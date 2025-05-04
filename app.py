# app.py

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

def predict_sentiment(text, pipe):
    output = pipe(text)[0]  # List of dicts for each label
    top = max(output, key=lambda x: x["score"])
    return top

def main():
    st.title("🌍 Multilingual Sentiment Analyzer")
    st.write("Using `cardiffnlp/twitter-xlm-roberta-base-sentiment`")

    pipe = load_model()

    st.subheader("Enter your text:")
    user_input = st.text_area("Type a sentence in any language (English, Spanish, Arabic, etc.):", height=150)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = predict_sentiment(user_input, pipe)
            st.success(f"Sentiment: **{result['label']}** — Confidence: **{result['score']:.2f}**")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
