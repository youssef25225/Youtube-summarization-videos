import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

CONFIDENCE_THRESHOLD = 0.1


@st.cache_resource
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")
    return tokenizer, model


def ask_question(question: str, transcript: str) -> str:
    tokenizer, model = load_qa_model()

    inputs = tokenizer(
        question,
        transcript,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )

    if not answer.strip() or answer.strip() in ("[CLS]", "[SEP]", ""):
        return "I couldn't find a confident answer in the video transcript."

    return answer
