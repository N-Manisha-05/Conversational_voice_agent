# tts.py — Deepgram TTS
import asyncio
import os
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
VOICE = "aura-asteria-en"
SAMPLE_RATE = 24000


async def speak_sentence(text: str):
    response = await client.speak.asyncrest.v("1").stream_memory(
        {"text": text},
        {"model": VOICE, "encoding": "linear16", "sample_rate": SAMPLE_RATE},
    )
    raw = response.stream.read()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767
    sd.play(audio, samplerate=SAMPLE_RATE)
    await asyncio.get_event_loop().run_in_executor(None, sd.wait)


async def synthesize_to_file(text: str, output_path: str = "output.wav") -> str:
    import wave
    response = await client.speak.asyncrest.v("1").stream_memory(
        {"text": text},
        {"model": VOICE, "encoding": "linear16", "sample_rate": SAMPLE_RATE},
    )
    raw = response.stream.read()
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw)
    return output_path


def stop_speaking():
    sd.stop()