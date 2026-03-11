import streamlit as st
from transformers import pipeline

CONFIDENCE_THRESHOLD = 0.1


@st.cache_resource
def load_qa_model():
    return pipeline(
        "text-classification",
        model="deepset/roberta-base-squad2",
        device=-1
    )


@st.cache_resource
def load_qa_pipeline():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch

    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    return tokenizer, model


def ask_question(question: str, transcript: str) -> str:
    tokenizer, model = load_qa_pipeline()
    import torch

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

    if not answer.strip() or answer.strip() == tokenizer.cls_token:
        return "I couldn't find a confident answer in the video transcript."

    return answer
