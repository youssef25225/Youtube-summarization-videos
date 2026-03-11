from transformers import pipeline

qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

CONFIDENCE_THRESHOLD = 0.1

def ask_question(question: str, transcript: str) -> str:
    """
    Answer a question based on the transcript.
    Uses the full transcript as context directly — no re-summarization.
    Returns a fallback message if confidence is too low.
    """
    result = qa_model(
        question=question,
        context=transcript
    )

    if result["score"] < CONFIDENCE_THRESHOLD:
        return "I couldn't find a confident answer in the video transcript."

    return result["answer"]
