from transformers import pipeline
import streamlit as st
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

CHUNK_SIZE = 500
MAX_LENGTH = 400
MIN_LENGTH = 150


@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1
    )


def _chunk_text(text: str):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())

        if current_words + words > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)
        current_words += words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summary(text: str):

    if not text or len(text.strip()) < 50:
        return {
            "points": [],
            "detailed": "Transcript too short."
        }

    summarizer = load_model()

    chunks = _chunk_text(text)

    chunk_summaries = []

    for chunk in chunks:

        result = summarizer(
            chunk,
            max_length=MAX_LENGTH,
            min_length=MIN_LENGTH,
            do_sample=True,
            temperature=0.7,
            truncation=True
        )

        chunk_summaries.append(result[0]["summary_text"])

    combined = " ".join(chunk_summaries)

    final_summary = summarizer(
        combined,
        max_length=500,
        min_length=200,
        do_sample=True,
        temperature=0.7,
        truncation=True
    )

    key_points = sent_tokenize(final_summary[0]["summary_text"])[:5]

    return {
        "points": key_points,
        "detailed": final_summary[0]["summary_text"],
        "chunks": chunk_summaries
    }


def display(data):

    st.subheader("Key Points")

    for p in data["points"]:
        st.write("•", p)

    st.subheader("Detailed Summary")
    st.write(data["detailed"])

    st.subheader("Section Summaries")

    for i, chunk in enumerate(data["chunks"], 1):
        st.write(f"**Part {i}:**")
        st.write(chunk)