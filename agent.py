import asyncio
import os
import sys
import re
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from groq import AsyncGroq
from livekit.agents.vad import VADEventType
from livekit.plugins.silero import VAD as SileroVAD
from livekit.rtc import AudioFrame
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from tts_piper import speak_sentence, stop_speaking

load_dotenv()





SAMPLE_RATE = 16000

CHUNK_SAMPLES = 480

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

SYSTEM_PROMPT = """You are a friendly voice assistant talking to someone in India.
Keep ALL responses under 3 sentences. Be casual, warm, and natural — like talking to a smart friend.
Use contractions. No bullet points. No markdown. Speak like a human, not a document."""

conversation_history = []
current_task: asyncio.Task | None = None
is_speaking = False
speaking_lock = asyncio.Lock()   # ADD THIS
last_interrupt_time = 0
INTERRUPT_COOLDOWN = 0.5


# ── LLM ───────────────────────────────────────────────────────────
async def ask_groq_streaming(user_text: str):
    conversation_history.append({"role": "user", "content": user_text})

    stream = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        temperature=0.8,
        max_tokens=150,
        stream=True,
    )

    buffer = ""
    full_reply = ""

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        full_reply += delta

        while re.search(r'[.!?]\s', buffer):
            match = re.search(r'[.!?]\s', buffer)
            sentence = buffer[:match.end()].strip()
            buffer = buffer[match.end():]
            if sentence:
                yield sentence

    if buffer.strip():
        yield buffer.strip()

    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)
    conversation_history.append({"role": "assistant", "content": full_reply.strip()})


# ── HANDLE ONE TURN ───────────────────────────────────────────────
async def handle_turn(transcript: str, show_user_text: bool = True):
    global is_speaking
    try:
        if not transcript.strip():
            return
        if show_user_text:
            print(f"You:  {transcript}")
        print("AI: ", end="", flush=True)
        async for sentence in ask_groq_streaming(transcript):
            print(sentence, end=" ", flush=True)
            await asyncio.sleep(0)
            async with speaking_lock:
                is_speaking = True
                await speak_sentence(sentence)
                is_speaking = False
        print()
    except asyncio.CancelledError:
        stop_speaking()
        is_speaking = False
    except Exception as e:
        print(f"\n[error] {e}")
        is_speaking = False


# ── MAIN ──────────────────────────────────────────────────────────
async def main():
    global current_task

    vad = SileroVAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.4,
        prefix_padding_duration=0.3,
        activation_threshold=0.3,
    )
    vad_stream = vad.stream()

    loop = asyncio.get_event_loop()
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()

    dg_live = dg_client.listen.websocket.v("1")

    def on_transcript(*args, **kwargs):
        result = kwargs.get("result") or (args[1] if len(args) > 1 else None)
        if result is None:
            return
        try:
            sentence = result.channel.alternatives[0].transcript
            is_final = result.is_final
            if sentence and is_final:
                loop.call_soon_threadsafe(transcript_queue.put_nowait, sentence)
        except Exception:
            pass

    dg_live.on(LiveTranscriptionEvents.Transcript, on_transcript)

    live_options = LiveOptions(
        model="nova-2",
        language="en-IN",
        smart_format=True,
        punctuate=True,
        interim_results=True,
        utterance_end_ms="1000",
        vad_events=False,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
    )

    await asyncio.to_thread(dg_live.start, live_options)
    print("Deepgram live STT connected.")

    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def audio_callback(indata, frames, time, status):
        chunk = indata[:, 0].copy()
        loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)

    async def keepalive():
        """Send Deepgram KeepAlive every 5s to prevent connection timeout."""
        while True:
            await asyncio.sleep(5)
            try:
                await asyncio.to_thread(dg_live.keep_alive)
            except Exception:
                pass

    async def send_silence_while_speaking():
        """Continuously feed silence to Deepgram while TTS is active,
        independent of the mic audio queue, to prevent timeout."""
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes()
        while True:
            await asyncio.sleep(0.05)  # 50ms interval
            if is_speaking:
                try:
                    await asyncio.to_thread(dg_live.send, silence)
                except Exception:
                    pass

    async def feed_audio():
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
        while True:
            chunk = await audio_queue.get()
            pcm_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)

            vad_stream.push_frame(AudioFrame(
                data=pcm_int16.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(pcm_int16),
            ))

            try:
                data_to_send = silence.tobytes() if is_speaking else pcm_int16.tobytes()
                await asyncio.to_thread(dg_live.send, data_to_send)
            except Exception as e:
                print("[Deepgram send error]", e)


    async def handle_vad_events():
        global current_task, last_interrupt_time, is_speaking

        async for event in vad_stream:
            if event.type == VADEventType.START_OF_SPEECH:
                if is_speaking:
                    now = asyncio.get_event_loop().time()
                    if now - last_interrupt_time > INTERRUPT_COOLDOWN:
                        # Clear is_speaking FIRST so feed_audio resumes
                        # sending real audio to Deepgram immediately
                        if current_task and not current_task.done():
                            current_task.cancel()
                        stop_speaking()
                        is_speaking = False
                        print("\n[barge-in detected]")
                        last_interrupt_time = now

    async def handle_transcripts():
        global current_task
        while True:
            transcript = await transcript_queue.get()
            if current_task and not current_task.done():
                current_task.cancel()
                stop_speaking()
            current_task = asyncio.create_task(handle_turn(transcript))

    print("Listening. Press Ctrl+C to stop.\n")

    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    )

    try:
        mic_stream.start()
        await asyncio.gather(
            feed_audio(),
            handle_vad_events(),
            handle_transcripts(),
            keepalive(),
            send_silence_while_speaking(),
        )
    finally:
        mic_stream.stop()
        mic_stream.close()
        stop_speaking()
        try:
            await asyncio.to_thread(dg_live.finish)
        except Exception:
            pass
        print("\nMic closed.")


# ── TEXT MODE ─────────────────────────────────────────────────────
async def text_mode():
    print("Text mode. Type your message. Type 'quit' to exit.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() == "quit":
            break
        if text:
            await handle_turn(text, show_user_text=False)


if __name__ == "__main__":
    if "--text" in sys.argv:
        try:
            asyncio.run(text_mode())
        except KeyboardInterrupt:
            print("\nGoodbye.")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nGoodbye.")