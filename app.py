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
    try:
        output = pipe(text)[0]  # List of dicts for each label
        top = max(output, key=lambda x: x["score"])
        return top
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        return None

def main():
    st.title("🌍 Multilingual Sentiment Analyzer")
    st.write("Using `cardiffnlp/twitter-xlm-roberta-base-sentiment`")

    try:
        pipe = load_model()
    except Exception as e:
        st.error(f"❌ Model Loading Error: {str(e)}")
        return  # Stop execution if model fails to load

    st.subheader("Enter your text:")
    user_input = st.text_area("Type a sentence in any language (English, Spanish, Arabic, etc.):", height=150)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            result = predict_sentiment(user_input, pipe)
            if result:
                st.success(f"Sentiment: **{result['label']}** — Confidence: **{result['score']:.2f}**")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
