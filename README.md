# 🎬 YouTube Video Summarizer

**Get YouTube video transcripts and summaries with one click!**  
This tool extracts audio from any YouTube video, transcribes it using high-speed Whisper models, and summarizes the content with a transformer-based NLP model.

🔗 **Try it live**: [ytsummaryweb.streamlit.app](https://ytsummaryweb.streamlit.app) <br />
📊 **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rishabh2007/youtube-transcript-summarizer-with-distilbert)

---

## ⚠️ Caution

> This app uses heavy transformer models for transcription and summarization.  
> ⏳ **It might take time** depending on video length, model size, and server resources.  
> 💤 For best results, avoid very long videos and retry if a timeout occurs.

---

## 🚀 Features

- 🔗 Input any YouTube video link
- 🎧 Downloads and processes audio using `yt-dlp` & `ffmpeg`
- 🗣️ Transcribes spoken content via `Faster-Whisper` (`tiny.en`)
- 🧠 Summarizes the entire video using Hugging Face’s `DistilBART`
- 📄 Provides:
  - Full transcript  
  - Clean summary (adjustable length)

---

## 🧪 Tech Stack

| Stage              | Tech Used                        |
|--------------------|----------------------------------|
| Audio Extraction   | `yt-dlp`, `ffmpeg`               |
| Transcription      | `Faster-Whisper` (tiny.en)       |
| Summarization      | `DistilBART` from Hugging Face   |
| Deployment         | `Streamlit`, `PyTorch`, `Transformers` |

---

## 📦 Setup Instructions

### Requirements

```bash
pip install -r requirements.txt
