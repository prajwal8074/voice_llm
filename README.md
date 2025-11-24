# Voice Support Bot

A voice-enabled customer support bot for managing support tickets.

## Features

- Voice and text interaction
- Create support tickets
- Cancel tickets by ID
- List open/closed tickets
- Local speech processing
- SQLite database storage

## Tech Stack

- **Speech Recognition**: Faster Whisper (local)
- **Text-to-Speech**: Piper TTS (local)  
- **AI Model**: Gemini 2.0 Flash
- **Web Interface**: Gradio
- **Database**: SQLite
- **Real-time Audio**: FastRTC

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
python3 -m piper.download_voices en_US-lessac-medium
```

2. Add Gemini API key to `.env` file

3. Run the app:
```bash
python app.py
```

## Usage

- Use microphone for voice input
- Click buttons for quick actions
- Type in text box for manual input

## Files

- `app.py` - Main application
- `functions.py` - AI and database functions
- `models.py` - Speech processing
- `requirements.txt` - Dependencies
