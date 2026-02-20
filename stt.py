# stt.py — kept for gradio/text mode compatibility
import io
import wave
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 480

dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

DG_OPTIONS = PrerecordedOptions(
    model="nova-2",
    language="en-IN",
    smart_format=True,
    punctuate=True,
)


async def transcribe_file(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    response = await dg_client.listen.asyncrest.v("1").transcribe_file(
        {"buffer": audio_bytes, "mimetype": "audio/wav"},
        DG_OPTIONS,
    )
    return response.results.channels[0].alternatives[0].transcript


async def transcribe_frames(frames) -> str:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for frame in frames:
            wf.writeframes(bytes(frame.data))
    response = await dg_client.listen.asyncrest.v("1").transcribe_file(
        {"buffer": buf.getvalue(), "mimetype": "audio/wav"},
        DG_OPTIONS,
    )
    return response.results.channels[0].alternatives[0].transcript