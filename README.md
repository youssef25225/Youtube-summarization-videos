# 🎬 Video Summarizer

An AI-powered Streamlit app that transcribes YouTube videos, summarizes them, and lets you ask questions about the content.

---

## Features

- **Transcription** — Downloads and transcribes YouTube videos using Whisper
- **Summarization** — Summarizes transcripts using BART (`facebook/bart-large-cnn`) with chunked processing for long videos
- **Q&A** — Ask questions about the video using a RoBERTa-based extractive QA model (`deepset/roberta-base-squad2`)
- **Chat history** — Full conversation history persisted within the session

---

## Project Structure

```
.
├── app.py              # Streamlit UI
├── summarizer.py       # Summarization logic (BART)
├── model.py            # Q&A logic (RoBERTa)
├── extract_text.py     # YouTube transcription (Whisper)
└── requirements.txt
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/video-summarizer.git
cd video-summarizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg (required by Whisper)

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

---

## Usage

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

1. Paste a YouTube link in the **Summary** tab
2. Click **Summarize Video** and wait for processing
3. Switch to the **Quiz** tab to ask questions about the video

---

## Models Used

| Task | Model | Source |
|------|-------|--------|
| Transcription | Whisper | OpenAI |
| Summarization | `facebook/bart-large-cnn` | Meta / Hugging Face |
| Question Answering | `deepset/roberta-base-squad2` | deepset / Hugging Face |

---

## Notes

- First run will download model weights (~1.6GB total). Subsequent runs use the cache.
- Long videos are chunked and summarized in passes — quality remains consistent regardless of transcript length.
- QA answers with a confidence score below `0.1` return a fallback message instead of a low-quality guess.
- GPU is not required but will significantly speed up transcription and summarization.

---

## Requirements

- Python 3.9+
- ffmpeg installed on system PATH
- Internet connection for first-time model downloads
