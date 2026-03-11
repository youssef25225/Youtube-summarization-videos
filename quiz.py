import streamlit as st
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/deepset/minilm-uncased-squad2"


def _get_headers():
    token = st.secrets["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"]
    return {"Authorization": f"Bearer {token}"}


def ask_question(question: str, transcript: str) -> str:
    headers = _get_headers()

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={
            "inputs": {
                "question": question,
                "context": transcript[:2000]  # trim to avoid payload limits
            }
        }
    )
    response.raise_for_status()
    result = response.json()

    if result.get("score", 0) < 0.1:
        return "I couldn't find a confident answer in the video transcript."

    return result["answer"]
