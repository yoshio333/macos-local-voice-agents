"""StickS3 thinking-companion WebSocket wire protocol serializer.

Wire format (matches the voicestick fork firmware, Phase B):
  - Binary WS frames = raw Opus packets (60ms / 16kHz / mono voice audio)
  - Text WS frames   = JSON control messages

Inbound (device -> bot):
  - {"event":"hello", ...}
  - {"event":"button","kind":"down|up|click","button":"primary|secondary",
     "duration_ms":N,"session_id":N}
Outbound (bot -> device):
  - {"event":"ui_state","state":"...","text":"..."}
  - {"event":"interaction_mode","mode":"hold_to_talk|click_to_talk"}

PTT turn-taking is button-driven (no VAD): primary button down -> the user
started speaking, up -> stopped. The firmware streams audio only while held.

Opus codec is handled via PyAV (bundled libopus). opuslib was avoided because
it relies on ctypes.util.find_library which cannot locate Homebrew's libopus,
and launchd strips DYLD_* env vars that would otherwise paper over it.
"""

import json
from fractions import Fraction
from typing import Optional, Union

import av
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

SAMPLE_RATE = 16000
FRAME_MS = 60


class OpusFrameSerializer(FrameSerializer):
    """Serializer for the StickS3 device WebSocket link."""

    def __init__(self):
        self._decoder = None
        self._resampler = None
        self._encoder = None
        self._enc_pts = 0

    @property
    def type(self) -> FrameSerializerType:
        # The transport sends str/bytes verbatim regardless of this value;
        # audio (binary) is the primary path.
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        self._init_codecs()

    def _init_codecs(self):
        if self._decoder is None:
            dec = av.CodecContext.create("libopus", "r")
            dec.sample_rate = SAMPLE_RATE
            dec.format = "s16"
            dec.layout = "mono"
            self._decoder = dec
            # libopus always decodes to 48kHz internally; resample down.
            self._resampler = av.AudioResampler(
                format="s16", layout="mono", rate=SAMPLE_RATE
            )
        if self._encoder is None:
            enc = av.CodecContext.create("libopus", "w")
            enc.sample_rate = SAMPLE_RATE
            enc.format = "s16"
            enc.layout = "mono"
            enc.options = {
                "frame_duration": str(FRAME_MS),
                "application": "voip",
                "b": "24000",
            }
            self._encoder = enc

    # ------------------------------------------------------------------
    # Outbound: bot -> device
    # ------------------------------------------------------------------
    async def serialize(self, frame: Frame) -> Optional[Union[str, bytes]]:
        if isinstance(frame, OutputAudioRawFrame):
            return self._encode_audio(frame)
        if isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            msg = frame.message
            return msg if isinstance(msg, str) else json.dumps(msg)
        return None

    def _encode_audio(self, frame: OutputAudioRawFrame) -> Optional[bytes]:
        self._init_codecs()
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        if pcm.size == 0:
            return None
        af = av.AudioFrame.from_ndarray(
            pcm.reshape(1, -1), format="s16", layout="mono"
        )
        af.sample_rate = SAMPLE_RATE
        af.pts = self._enc_pts
        af.time_base = Fraction(1, SAMPLE_RATE)
        self._enc_pts += pcm.size
        try:
            packets = self._encoder.encode(af)
        except Exception as e:
            logger.error(f"Opus encode failed: {e}")
            return None
        if not packets:
            return None
        if len(packets) > 1:
            # A 60ms output chunk should yield exactly one 60ms Opus packet.
            logger.warning(
                f"OpusFrameSerializer: {len(packets)} packets from one chunk; "
                "sending only the first"
            )
        return bytes(packets[0])

    # ------------------------------------------------------------------
    # Inbound: device -> bot
    # ------------------------------------------------------------------
    async def deserialize(self, data: Union[str, bytes]) -> Optional[Frame]:
        if isinstance(data, (bytes, bytearray)):
            return self._decode_audio(bytes(data))
        return self._handle_control(data)

    def _decode_audio(self, data: bytes) -> Optional[InputAudioRawFrame]:
        self._init_codecs()
        try:
            frames = self._decoder.decode(av.Packet(data))
        except Exception as e:
            logger.error(f"Opus decode failed: {e}")
            return None
        chunks = []
        for f in frames:
            for rf in self._resampler.resample(f):
                chunks.append(rf.to_ndarray())
        if not chunks:
            return None
        pcm = np.concatenate(chunks, axis=1).astype(np.int16)
        return InputAudioRawFrame(
            audio=pcm.tobytes(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

    def _handle_control(self, data: str) -> Optional[Frame]:
        try:
            msg = json.loads(data)
        except (ValueError, TypeError):
            logger.warning(f"OpusFrameSerializer: bad control JSON: {data!r}")
            return None
        event = msg.get("event")
        if event == "hello":
            logger.info(f"device hello: {msg}")
            return None
        if event == "button":
            return self._handle_button(msg)
        logger.debug(f"unhandled control event: {event!r}")
        return None

    def _handle_button(self, msg: dict) -> Optional[Frame]:
        button = msg.get("button")
        kind = msg.get("kind")
        # Only the primary button drives PTT turn-taking. Secondary button and
        # primary "click" (a tap, not a hold) are reserved for later phases.
        if button != "primary":
            logger.debug(f"ignoring {button!r} button {kind!r}")
            return None
        if kind == "down":
            logger.info("PTT down -> UserStartedSpeaking")
            return UserStartedSpeakingFrame()
        if kind == "up":
            logger.info("PTT up -> UserStoppedSpeaking")
            return UserStoppedSpeakingFrame()
        return None
