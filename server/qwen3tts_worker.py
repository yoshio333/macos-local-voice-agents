#!/usr/bin/env python3
"""
Standalone Qwen3-TTS worker process.

Communicates via JSON over stdin/stdout. Mirrors kokoro_worker.py structure.

Commands:
    {"cmd": "init", "model": "...", "voice": "eric", "language": "Japanese", "speed": 1.4}
    {"cmd": "generate", "text": "..."}                    # 一括（後方互換）
    {"cmd": "generate_stream", "text": "...", "interval": 0.32}
        → 0個以上の {"chunk": "<b64>", "more": true} を返したあと
          {"success": true, "more": false} で終端
"""

import sys
import os
import json
import base64
import logging
import numpy as np

_real_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")

logging.basicConfig(level=logging.INFO, format="QWEN3TTS-WORKER: %(message)s", stream=sys.stderr)


def _reply(obj):
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class Worker:
    def __init__(self):
        self.model = None
        self.speaker = None
        self.language = None
        self.speed = 1.0

    def _apply_speed(self, audio):
        if abs(self.speed - 1.0) < 1e-3:
            return audio
        from pedalboard import time_stretch
        out = time_stretch(audio.astype(np.float32), samplerate=24000, stretch_factor=self.speed)
        if out.ndim == 2:
            out = out[0]
        return out.astype(np.float32)

    def _to_int16_b64(self, audio):
        if np.max(np.abs(audio)) < 1e-6:
            return None
        audio_int16 = (audio * 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode()

    def initialize(self, model_name, voice, language, speed=1.0):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.speaker = voice
            self.language = language or "Japanese"
            self.speed = float(speed) if speed else 1.0
            list(self.model.generate_custom_voice(
                text="テスト",
                speaker=self.speaker,
                language=self.language,
            ))
            return {"success": True}
        except Exception as e:
            import traceback
            return {"error": f"{e}\n{traceback.format_exc()}"}

    def generate(self, text):
        """Non-streaming, full-audio generation. Backward compat."""
        try:
            if not self.model:
                return {"error": "Not initialized"}

            segments = []
            for result in self.model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
            ):
                audio_data = np.array(result.audio, copy=True)
                segments.append(audio_data)

            if not segments:
                return {"error": "No audio"}

            audio = segments[0] if len(segments) == 1 else np.concatenate(segments, axis=0)
            audio = self._apply_speed(audio)
            b64 = self._to_int16_b64(audio)
            if b64 is None:
                return {"error": "Generated audio is silent"}
            return {"success": True, "audio": b64}
        except Exception as e:
            import traceback
            return {"error": f"{e}\n{traceback.format_exc()}"}

    def generate_stream(self, text, interval=0.32):
        try:
            if not self.model:
                _reply({"error": "Not initialized", "more": False})
                return

            chunk_count = 0
            for result in self.model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
                stream=True,
                streaming_interval=float(interval),
            ):
                audio = np.array(result.audio, copy=True)
                audio = self._apply_speed(audio)
                b64 = self._to_int16_b64(audio)
                if b64 is None:
                    continue
                _reply({"chunk": b64, "more": True})
                chunk_count += 1
                print(f"streamed chunk {chunk_count} ({len(b64)} b64 chars)", file=sys.stderr)

            _reply({"success": True, "more": False, "chunks": chunk_count})
        except Exception as e:
            import traceback
            _reply({"error": f"{e}\n{traceback.format_exc()}", "more": False})


def main():
    worker = Worker()
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            cmd = req.get("cmd")
            if cmd == "init":
                resp = worker.initialize(
                    req["model"],
                    req["voice"],
                    req.get("language", "Japanese"),
                    req.get("speed", 1.0),
                )
                _reply(resp)
            elif cmd == "generate":
                resp = worker.generate(req["text"])
                _reply(resp)
            elif cmd == "generate_stream":
                worker.generate_stream(req["text"], req.get("interval", 0.32))
            else:
                _reply({"error": "Unknown command"})
        except Exception as e:
            _reply({"error": str(e), "more": False})


if __name__ == "__main__":
    main()
