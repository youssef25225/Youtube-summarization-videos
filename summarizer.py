import streamlit as st
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
CHUNK_SIZE = 900


def _get_headers():
    token = st.secrets["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"]
    return {"Authorization": f"Bearer {token}"}


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
    return chunks


def summary(text: str) -> str:
    headers = _get_headers()
    chunks = _chunk_text(text)

    chunk_summaries = []
    for chunk in chunks:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": chunk, "parameters": {"max_length": 130, "min_length": 30}}
        )
        response.raise_for_status()
        result = response.json()
        chunk_summaries.append(result[0]["summary_text"])

    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": combined, "parameters": {"max_length": 130, "min_length": 30}}
        )
        response.raise_for_status()
        return response.json()[0]["summary_text"]

    return chunk_summaries[0]
