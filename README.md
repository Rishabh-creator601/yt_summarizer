# 🎬 YouTube Video Summarizer

**Get YouTube video transcripts and summaries with one click!**  
This tool extracts audio from any YouTube video, transcribes it using high-speed Whisper models, and summarizes the content with a transformer-based NLP model.

🔗 **Try it live**: [ytsummaryweb.streamlit.app](https://ytsummaryweb.streamlit.app)
📊 **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rishabh2007/youtube-transcript-summarizer-with-distilbert)

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
