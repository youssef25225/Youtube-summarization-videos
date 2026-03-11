import streamlit as st
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

CHUNK_SIZE = 900
MAX_LENGTH = 180
MIN_LENGTH = 40


@st.cache_resource
def load_summarizer():
    return pipeline(
        "text2text-generation",
        model="facebook/bart-large-cnn",
        device=-1
    )


def _chunk_text(text: str) -> list[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summary(text: str) -> str:
    model = load_summarizer()
    chunks = _chunk_text(text)

    chunk_summaries = []
    for chunk in chunks:
        result = model(
            chunk,
            max_new_tokens=MAX_LENGTH,
            min_new_tokens=MIN_LENGTH,
        )
        chunk_summaries.append(result[0]["generated_text"])

    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        final = model(
            combined,
            max_new_tokens=MAX_LENGTH,
            min_new_tokens=MIN_LENGTH,
        )
        return final[0]["generated_text"]

    return chunk_summaries[0]
