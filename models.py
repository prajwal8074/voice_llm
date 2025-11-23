import requests
import os

# Speech To Text
from faster_whisper import WhisperModel

from openai import OpenAI

# Text To Speech
from piper import PiperVoice

import wave
import pyaudio
import numpy as np
from dotenv import load_dotenv

import gradio as gr

load_dotenv()

stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

pyAudio = pyaudio.PyAudio()

voice = PiperVoice.load("en_US-lessac-medium.onnx")

def transcribe(audio_filename):
    segments, info = stt_model.transcribe(audio_filename, beam_size=1)
    full_text = "".join([segment.text for segment in segments])
    return full_text

def get_response(user_text):
    return client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
        {"role": "system", "content": "Answer user's all queries in one line."},
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": user_text,
            }
          ],
        }
      ],
    ).choices[0].message.content

def synthesize_audio(text):
    audio_buffer = bytearray()
    sample_rate = None
    
    for chunk in voice.synthesize(text):
        audio_buffer.extend(chunk.audio_int16_bytes)
        
    audio_np = np.frombuffer(audio_buffer, dtype=np.int16)
    
    sample_rate = voice.config.sample_rate
    
    return sample_rate, audio_np
