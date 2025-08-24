from flask import Flask, render_template, request, jsonify, url_for
import os
import yt_dlp
import subprocess
from datetime import datetime
import time
from groq import Groq
import logging
import requests
import glob
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# === Flask Setup ===
flask_app = Flask(__name__, template_folder="templates", static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static')))
AUDIO_FILES_DIR = os.path.join(flask_app.static_folder, "audio_files")
CHUNKS_DIR = os.path.join(AUDIO_FILES_DIR, "chunks")
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Replace this

# === Ollama Setup ===
llm = ChatOllama(model="llama3")  # Assumes 'llama3' model is installed and Ollama is running locally

# === Cleanup Function ===
def cleanup_old_files(directory, max_age_hours=24):
    now = time.time()
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_hours * 3600:
            os.remove(path)
            logger.info(f"[🗑] Deleted old file: {file}")

# === YouTube Download ===
def download_youtube_wav(url: str, max_retries=3) -> str | None:
    cleanup_old_files(AUDIO_FILES_DIR)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"audio_{timestamp}"
    output_path = os.path.join(AUDIO_FILES_DIR, f"{filename}.wav")

    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': ['-ac', '1', '-ar', '16000'],
        'outtmpl': os.path.join(AUDIO_FILES_DIR, f'{filename}.%(ext)s'),
        'restrictfilenames': True,
        'noplaylist': True,
        'quiet': False
    }

    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.extract_info(url, download=True)
            if os.path.exists(output_path):
                return output_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            time.sleep(2 ** attempt)
    return None

# === Split WAV File ===
def split_audio_to_chunks(audio_path, chunk_duration=600):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_pattern = os.path.join(CHUNKS_DIR, f"{base_name}_%03d.wav")

    command = [
        "ffmpeg", "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-c", "copy",
        output_pattern
    ]

    try:
        subprocess.run(command, check=True)
        return sorted(glob.glob(os.path.join(CHUNKS_DIR, f"{base_name}_*.wav")))
    except Exception as e:
        logger.error(f"Audio splitting failed: {e}")
        return []

# === Transcribe Chunk ===
def transcribe_with_groq_whisper(audio_path):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    try:
        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "audio/wav")
            }
            data = {
                "model": "whisper-large-v3"
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["text"]
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        return ""

# === Merge Transcripts ===
def transcribe_all_chunks(chunk_paths):
    full_transcript = ""
    for idx, path in enumerate(chunk_paths):
        logger.info(f"Transcribing chunk {idx+1}/{len(chunk_paths)}: {path}")
        text = transcribe_with_groq_whisper(path)
        full_transcript += f"\n{text}"
    return full_transcript.strip()

# === Summarize ===
def summarize_with_ollama(text):
    if not text or len(text.strip()) < 50:
        return "Text too short to summarize."
    try:
        messages = [
            SystemMessage(content="Summarize the following text:"),
            HumanMessage(content=text)
        ]
        response = llm.invoke(messages)
        summary = response.content.strip() if response.content else "Summarization failed, no content returned."
        return summary
    except Exception as e:
        logger.error(f"Error summarizing with Ollama: {e}")
        return f"Summarization failed: {e}"

# === Routes ===

@flask_app.route('/')
def index():
    return render_template("index.html")

@flask_app.route('/extract', methods=['POST'])
def extract_audio():
    try:
        data = request.get_json()
        video_url = data.get("video_url", "")
        if not video_url:
            return jsonify({"status": "error", "message": "No video URL provided", "audio_url": "", "transcript": ""})

        audio_path = download_youtube_wav(video_url)
        if not audio_path:
            return jsonify({"status": "error", "message": "Download failed", "audio_url": "", "transcript": ""})

        chunk_paths = split_audio_to_chunks(audio_path)
        if not chunk_paths:
            return jsonify({"status": "error", "message": "Audio splitting failed", "audio_url": "", "transcript": ""})

        transcript = transcribe_all_chunks(chunk_paths)
        if not transcript:
            return jupytext({"status": "error", "message": "Transcription failed", "audio_url": "", "transcript": ""})

        audio_url = url_for('static', filename=f"audio_files/{os.path.basename(audio_path)}")
        return jsonify({"status": "success", "audio_url": audio_url, "transcript": transcript})
    except Exception as e:
        logger.error(f"Extract error: {e}")
        return jsonify({"status": "error", "message": str(e), "audio_url": "", "transcript": ""})

@flask_app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    try:
        data = request.get_json()
        transcript = data.get("transcript", "")
        if not transcript:
            return jsonify({"status": "error", "message": "No transcript provided", "summary": ""})

        summary = summarize_with_ollama(transcript)
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return jsonify({"status": "error", "message": str(e), "summary": ""})

@flask_app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        question = data.get("question", "")
        transcript = data.get("transcript", "")

        if not question or not transcript:
            return jsonify({"status": "error", "message": "Missing question or transcript", "answer": ""})

        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
You are a highly knowledgeable assistant. Use the full transcript below to answer the user's question.

Transcript:
{transcript}

User Question:
{question}
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return jsonify({"status": "success", "answer": response.choices[0].message.content})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"status": "error", "message": str(e), "answer": ""})

# === Run App ===
if __name__ == '__main__':
    logger.info("🚀 Starting Flask app at http://127.0.0.1:5001")
    flask_app.run(debug=True, port=5001)