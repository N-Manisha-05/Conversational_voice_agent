import asyncio
import os
import threading
import numpy as np
import sounddevice as sd

# Configuration
PIPER_BINARY = os.path.abspath("./piper/piper")
MODEL_PATH = os.path.abspath("./piper/en_US-lessac-medium.onnx")
SAMPLE_RATE = 22050  # Piper uses 22050Hz for this model
PLAYBACK_BLOCK = 1024  # frames per write to the output stream

# Global stop event for interrupting playback
_stop_event = threading.Event()


def _play_audio_blocking(audio_data: np.ndarray):
    """Play audio using a dedicated OutputStream (avoids conflicts with InputStream)."""
    _stop_event.clear()
    idx = 0
    total = len(audio_data)

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while idx < total and not _stop_event.is_set():
            end = min(idx + PLAYBACK_BLOCK, total)
            chunk = audio_data[idx:end]
            stream.write(chunk.reshape(-1, 1))
            idx = end


async def speak_sentence(text: str):
    """
    Synthesizes speech using Piper and plays it immediately.
    """
    if not text.strip():
        return

    proc = await asyncio.create_subprocess_exec(
        PIPER_BINARY,
        "--model", MODEL_PATH,
        "--output-raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    if proc.stdin:
        try:
            proc.stdin.write(text.encode('utf-8'))
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    raw_audio = b""
    if proc.stdout:
        raw_audio = await proc.stdout.read()

    await proc.wait()

    if raw_audio:
        audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32767.0

        loop = asyncio.get_event_loop()
        try:
            await asyncio.shield(
                loop.run_in_executor(None, _play_audio_blocking, audio_data)
            )
        except asyncio.CancelledError:
            _stop_event.set()
            raise


def stop_speaking():
    """Signal the playback thread to stop immediately."""
    _stop_event.set()
