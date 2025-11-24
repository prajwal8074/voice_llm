# Voice Support Bot

A voice-enabled customer support bot for managing support tickets.
<img width="1457" height="777" alt="Screenshot from 2025-11-24 22-24-44" src="https://github.com/user-attachments/assets/23f2c562-b241-4022-b8fb-279b74b94368" />


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

## Requirements

- Gemini API Key
- Twillio Account

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
python3 -m piper.download_voices en_US-lessac-medium
```

2. create .env file with contents:
```
GEMINI_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
```

3. Run the app:
```bash
python3 app.py
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
