import streamlit as st
import extract_text
import summarizer
import quiz

st.title("Video Summarizer")

if "text" not in st.session_state:
    st.session_state.text = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

t1, t2 = st.tabs(["Summary", "Ask"])

with t1:
    link = st.text_input("Enter YouTube link")

    if st.button("Summarize Video"):
        if not link.strip():
            st.warning("Please enter a valid YouTube link.")
        else:
            try:
                with st.spinner("Processing video..."):
                    text = extract_text.transcribe(link)
                    summary = summarizer.summary(text)
                    st.session_state.text = text
                    st.session_state.summary = summary
                    st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.summary:
        st.markdown("### Summary")
        st.write(st.session_state.summary)

with t2:
    if st.session_state.text is None:
        st.info("Please summarize a video first.")
    else:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])

        question = st.chat_input("Ask a question about the video")

        if question:
            try:
                with st.spinner("Generating answer..."):
                    answer = quiz.ask_question(
                        question,
                        st.session_state.text
                    )
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate answer: {e}")
