# Local Speech To Text
from faster_whisper import WhisperModel

# Local Text To Speech
from piper import PiperVoice

import numpy as np
from dotenv import load_dotenv

load_dotenv()

stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

voice = PiperVoice.load("en_US-lessac-medium.onnx")

def transcribe(audio_filename):
    segments, info = stt_model.transcribe(audio_filename, beam_size=1)
    full_text = "".join([segment.text for segment in segments])
    return full_text

def synthesize_audio(text):
    audio_buffer = bytearray()
    sample_rate = None
    
    for chunk in voice.synthesize(text):
        audio_buffer.extend(chunk.audio_int16_bytes)
        
    audio_np = np.frombuffer(audio_buffer, dtype=np.int16)
    
    sample_rate = voice.config.sample_rate
    
    return sample_rate, audio_np
