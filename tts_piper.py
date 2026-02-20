import asyncio
import os
import numpy as np
import sounddevice as sd

# Configuration
PIPER_BINARY = os.path.abspath("./piper/piper")
MODEL_PATH = os.path.abspath("./piper/en_US-lessac-medium.onnx")
SAMPLE_RATE = 22050  # Piper uses 22050Hz for this model


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

        sd.play(audio_data, samplerate=SAMPLE_RATE)

        loop = asyncio.get_event_loop()
        fut = loop.run_in_executor(None, sd.wait)

        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            sd.stop()
            raise


def stop_speaking():
    try:
        sd.stop()
    except Exception:
        pass
