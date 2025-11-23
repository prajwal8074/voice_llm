import gradio as gr
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions, AdditionalOutputs

import numpy as np
import scipy.io.wavfile as wavfile
import uuid
import os

import models

os.makedirs("saved_audio", exist_ok=True)

def save_audio_chunk(sample_rate, audio_data):
    filename = f"saved_audio/user_{uuid.uuid4()}.wav"
    data_to_save = audio_data.T
    wavfile.write(filename, sample_rate, data_to_save)
    return filename

def process_interaction(audio: tuple[int, np.ndarray], old_history, new_history):
    sample_rate, audio_data = audio
    audio_path = save_audio_chunk(sample_rate, audio_data)
    
    history = new_history
    if history is None:
        history = []
    
    user_msg = gr.ChatMessage(
        role="user",
        content=models.transcribe(audio_path)
    )
    
    history.append(user_msg)
    bot_msg = gr.ChatMessage(role="assistant", content="...")
    history.append(bot_msg)
    
    yield None, AdditionalOutputs(history)
    
    response_text = models.get_response(user_msg.content)
    
    history[-1].content = response_text
    output_audio = models.synthesize_audio(response_text)
    
    yield output_audio, AdditionalOutputs(history)
    
def update_chat_ui(old_history, new_history):
    return new_history

with gr.Blocks() as demo:
    
    with gr.Group():
        chatbot = gr.Chatbot(
            placeholder="<strong>Your Personal Yes-Man</strong><br>Speak, pause, and I will answer.",
            type="messages",
            render=False,
            elem_id="chatbot"
        )
        
        stream = Stream(
            modality="audio",
            mode="send-receive",
            handler=ReplyOnPause(
                process_interaction,
                algo_options=AlgoOptions(speech_threshold=0.3),
                model_options=SileroVadOptions(min_silence_duration_ms=500, threshold=0.3)
            ),
            additional_inputs=[chatbot],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chat_ui
        )
        
        stream.ui
        
        with gr.Row():
            gr.Markdown(
                "<h2 style='text-align: center; color: #4F8A10;'>💡 Try saying: 'What is the weather like today?' or 'Tell me a joke.'</h2>"
            )

if __name__ == "__main__":
    demo.launch()
