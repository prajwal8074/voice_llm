import gradio as gr
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions, AdditionalOutputs

import numpy as np
import scipy.io.wavfile as wavfile
import uuid
import os

import models
import functions

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
    
    response_text = functions.get_response(user_msg.content)
    
    history[-1].content = response_text
    output_audio = models.synthesize_audio(response_text)
    
    yield output_audio, AdditionalOutputs(history)

def process_text_input(text, history):
    history = history or []
    
    history.append(gr.ChatMessage(role="user", content=text))
    yield history, None
    
    history.append(gr.ChatMessage(role="assistant", content="..."))
    yield history, None
    
    response_text = functions.get_response(text)
    history[-1].content = response_text
    
    output_audio = models.synthesize_audio(response_text)
    
    yield history, output_audio
    
def update_chat_ui(old_history, new_history):
    return new_history

with gr.Blocks() as demo:
    
    with gr.Group():
        chatbot = gr.Chatbot(
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
        
        audio_player = gr.Audio(visible=True, autoplay=True)
        
        with gr.Row():
            btn_weather = gr.Button("Create a support ticket")
            btn_joke = gr.Button("Cancel a support ticket")
            btn_fact = gr.Button("List open tickets")
    
        def click_weather(history):
            yield from process_text_input("Create a support ticket", history)
            
        def click_joke(history):
            yield from process_text_input("Cancel a support ticket", history)
            
        def click_fact(history):
            yield from process_text_input("List open tickets", history)
        
        btn_weather.click(click_weather, inputs=[chatbot], outputs=[chatbot, audio_player])
        btn_joke.click(click_joke, inputs=[chatbot], outputs=[chatbot, audio_player])
        btn_fact.click(click_fact, inputs=[chatbot], outputs=[chatbot, audio_player])
        
        with gr.Row():
            txt_input = gr.Textbox(
                show_label=False,
                placeholder="Type your own message here...",
                scale=4,
                container=False
            )
            btn_send = gr.Button("Send", variant="primary", scale=1)

        def submit_custom(text, history):
            yield from process_text_input(text, history)

        txt_input.submit(submit_custom, [txt_input, chatbot], [chatbot, audio_player]).then(
            lambda: "", None, [txt_input]
        )
        btn_send.click(submit_custom, [txt_input, chatbot], [chatbot, audio_player]).then(
            lambda: "", None, [txt_input]
        )

if __name__ == "__main__":
    demo.launch(share=True)
