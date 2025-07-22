import streamlit as st
import subprocess, whisper, tqdm, os, math, warnings, torch
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
from transformers import BartTokenizer, BartForConditionalGeneration

warnings.filterwarnings("ignore")

# ============================
# CACHED MODELS
# ============================



st.title("🎧 YouTube Video  Summarizer")

url = st.text_input("Enter YouTube URL")


holder = st.empty()

@st.cache_resource
def load_whisper_model():
    if torch.cuda.is_available():
        model = WhisperModel("tiny.en", device="cuda", compute_type="float16")
    else:
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return model

@st.cache_resource
def load_summarization_model():
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return tokenizer, model

# ============================
# DOWNLOAD AUDIO
# ============================

def download_audio_with_ytdlp(url, output_path="audio.%(ext)s"):
    
    
    def progress_hook(d):
        if d['status'] == 'downloading':
            holder.write(f"Downloading: {d.get('_percent_str', '').strip()} | Speed: {d.get('_speed_str', '').strip()} | ETA: {d.get('_eta_str', '').strip()}")
        elif d['status'] == 'finished':
            holder.success(f"Download complete: {d['filename']}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        'progress_hooks': [progress_hook],
        'quiet': True,
        'noplaylist': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    duration_sec = info.get('duration')
    minutes, seconds = duration_sec // 60, duration_sec % 60
    st.info(f"🔊 Audio duration: {int(minutes)}m {int(seconds)}s")

    return output_path.replace('%(ext)s', 'mp3')


# ============================
# AUDIO TO TEXT
# ============================

def audio_to_text(path, model):
    segments, _ = model.transcribe(language="en", audio=path, chunk_length=120, temperature=0)

    final_text = ""
    for i, seg in enumerate(segments):
        final_text += seg.text + "\n"

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    return final_text



# ============================
# SUMMARIZE
# ============================

def summarize_text(text, tokenizer, model, max_len=500, min_len=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_len, min_length=min_len, length_penalty=2.0, num_beams=7, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# ============================
# STREAMLIT UI
# ============================




if url:
    with st.spinner("Downloading & Processing..."):
        whisper_model = load_whisper_model()
        tokenizer, bart_model = load_summarization_model()

        audio_path = download_audio_with_ytdlp(url)
        transcript = audio_to_text(audio_path, whisper_model)

        summary = summarize_text(transcript, tokenizer, bart_model)
        st.subheader("Summary")
        st.write(summary)
