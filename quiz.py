import streamlit as st
from transformers import pipeline

CONFIDENCE_THRESHOLD = 0.1


@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )


def ask_question(question: str, transcript: str) -> str:
    qa_model = load_qa_model()

    result = qa_model(
        question=question,
        context=transcript
    )

    if result["score"] < CONFIDENCE_THRESHOLD:
        return "I couldn't find a confident answer in the video transcript."

    return result["answer"]
