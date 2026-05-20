#
# Process-isolated MLX TTS service (Kokoro / Marvis / Qwen3-TTS)
# Uses a separate process to avoid Metal threading conflicts on Apple Silicon
#

import asyncio
import subprocess
import json
import base64
import sys
from typing import AsyncGenerator, Optional
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class TTSMLXIsolated(TTSService):
    """Isolated MLX TTS using subprocess to avoid Metal issues.

    Streaming path:
        For Qwen3-TTS models (which support streaming generation), the worker
        is invoked with `generate_stream` and audio is yielded chunk-by-chunk
        as it's produced. First-chunk latency drops dramatically on long text.
        Kokoro/Marvis still use the original one-shot `generate` command.
    """

    def __init__(
        self,
        *,
        model: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        language: Optional[str] = None,
        speed: float = 1.0,
        device: Optional[str] = None,
        sample_rate: int = 24000,
        streaming_interval: float = 0.32,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._language = language
        self._speed = speed
        self._device = device
        self._streaming_interval = streaming_interval
        self._supports_streaming = "Qwen3-TTS" in model

        self._process = None
        self._initialized = False

        self._worker_script = self._get_worker_script_path()

        self._settings = {
            "model": model,
            "voice": voice,
            "language": language,
            "sample_rate": sample_rate,
        }

    def _get_worker_script_path(self) -> str:
        current_dir = Path(__file__).parent
        if self._model_name.startswith("Marvis-AI"):
            worker_path = current_dir / "marvis_worker.py"
        elif "Qwen3-TTS" in self._model_name:
            worker_path = current_dir / "qwen3tts_worker.py"
        else:
            worker_path = current_dir / "kokoro_worker.py"

        logger.info(f"Using worker script: {worker_path}")

        if not worker_path.exists():
            raise FileNotFoundError(
                f"Worker script not found at {worker_path}. "
                "Make sure worker script is in the same directory as tts_mlx_isolated.py"
            )

        return str(worker_path)

    def _start_worker(self):
        try:
            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=0,
            )
            logger.info(f"Started {self._model_name} worker process: {self._process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            return False

    def _ensure_worker(self) -> bool:
        if not self._process or self._process.poll() is not None:
            logger.debug("Starting worker process...")
            return self._start_worker()
        return True

    def _write_command(self, command: dict) -> bool:
        try:
            self._process.stdin.write(json.dumps(command) + "\n")
            self._process.stdin.flush()
            return True
        except Exception as e:
            logger.error(f"Worker write error: {e}")
            return False

    def _read_line_blocking(self, timeout: float = 30.0) -> Optional[str]:
        """Read a single response line with timeout. Returns None on timeout/EOF."""
        import select

        ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not ready:
            return None
        line = self._process.stdout.readline()
        if not line:
            return None
        return line

    def _send_command(self, command: dict) -> dict:
        """Send a command expecting a single-line JSON response."""
        try:
            if not self._ensure_worker():
                return {"error": "Failed to start worker"}
            if not self._write_command(command):
                return {"error": "Failed to write command"}

            line = self._read_line_blocking(timeout=30.0)
            if line is None:
                if self._process.poll() is not None:
                    return {"error": "Worker process died"}
                return {"error": "Worker response timeout"}

            data = json.loads(line.strip())
            if "audio" in data:
                logger.debug(f"Worker response: success, {len(data['audio'])} chars audio")
            else:
                logger.debug(f"Worker response: {line.strip()[:200]}")
            return data
        except Exception as e:
            logger.error(f"Worker communication error: {e}")
            return {"error": str(e)}

    async def _initialize_if_needed(self):
        if self._initialized:
            return True

        loop = asyncio.get_event_loop()
        init_cmd = {"cmd": "init", "model": self._model_name, "voice": self._voice}
        if self._language is not None:
            init_cmd["language"] = self._language
        if self._speed != 1.0:
            init_cmd["speed"] = self._speed
        result = await loop.run_in_executor(None, self._send_command, init_cmd)

        if result.get("success"):
            self._initialized = True
            logger.info(f"{self._model_name} worker initialized (streaming={self._supports_streaming})")
            return True
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Worker initialization failed: {error_msg}")
            return False

    def can_generate_metrics(self) -> bool:
        return True

    async def _yield_audio_bytes(self, audio_bytes: bytes) -> AsyncGenerator[Frame, None]:
        CHUNK_SIZE = self.chunk_size
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            chunk = audio_bytes[i : i + CHUNK_SIZE]
            if len(chunk) > 0:
                yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                await asyncio.sleep(0.001)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            if not await self._initialize_if_needed():
                raise RuntimeError(f"Failed to initialize {self._model_name} worker")

            loop = asyncio.get_event_loop()

            if self._supports_streaming:
                # ----- Streaming path (Qwen3-TTS) -----
                cmd = {
                    "cmd": "generate_stream",
                    "text": text,
                    "interval": self._streaming_interval,
                }

                def _start_stream():
                    if not self._ensure_worker():
                        return False
                    return self._write_command(cmd)

                if not await loop.run_in_executor(None, _start_stream):
                    raise RuntimeError("Failed to start streaming generation")

                first_chunk = True
                total_chunks = 0
                while True:
                    line = await loop.run_in_executor(None, self._read_line_blocking, 60.0)
                    if line is None:
                        raise RuntimeError("Worker streaming timeout / EOF")

                    try:
                        msg = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logger.error(f"Bad worker line: {line.strip()[:200]!r}")
                        raise RuntimeError(f"Worker JSON parse error: {e}")

                    if msg.get("error"):
                        raise RuntimeError(f"Worker error: {msg['error']}")

                    chunk_b64 = msg.get("chunk")
                    if chunk_b64:
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False
                        audio_bytes = base64.b64decode(chunk_b64)
                        async for frame in self._yield_audio_bytes(audio_bytes):
                            yield frame
                        total_chunks += 1

                    if not msg.get("more", False):
                        logger.debug(
                            f"{self}: streamed {total_chunks} chunks for [{text[:30]}...]"
                        )
                        break
            else:
                # ----- One-shot path (Kokoro / Marvis) -----
                result = await loop.run_in_executor(
                    None, self._send_command, {"cmd": "generate", "text": text}
                )

                if not result.get("success"):
                    raise RuntimeError(f"Audio generation failed: {result.get('error')}")

                audio_bytes = base64.b64decode(result["audio"])
                await self.stop_ttfb_metrics()
                async for frame in self._yield_audio_bytes(audio_bytes):
                    yield frame

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _cleanup(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    async def __aenter__(self):
        await super().__aenter__()
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)
