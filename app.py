# app.py

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt

# 💾 Cache model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

# 🔍 Predict sentiment
def predict_sentiment(text, pipe):
    try:
        scores = pipe(text)[0]
        top = max(scores, key=lambda x: x["score"])
        return top, scores
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        return None, None

# 📊 Bar chart
def plot_bar_chart(scores):
    df = pd.DataFrame(scores)
    fig, ax = plt.subplots()
    ax.bar(df["label"], df["score"], color=["red", "gray", "green"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("📊 Sentiment Scores")
    st.pyplot(fig)

# 🥧 Pie chart
def plot_pie_chart(scores):
    df = pd.DataFrame(scores)
    fig, ax = plt.subplots()
    ax.pie(df["score"], labels=df["label"], autopct="%1.1f%%", colors=["red", "gray", "green"])
    ax.set_title("🧩 Sentiment Breakdown")
    st.pyplot(fig)

# 🚀 Main App
def main():
    st.set_page_config(page_title="Hustle's Multilingual Sentiment Analyzer", page_icon="🌍", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #00A36C;'>🌍 Multilingual Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("##### Built with 🤗 Transformers Model)

    st.divider()
    try:
        pipe = load_model()
    except Exception as e:
        st.error(f"❌ Model Loading Error: {str(e)}")
        return

    st.markdown("### ✍️ Enter text in any language:")
    user_input = st.text_area("Your text goes here...", height=150, label_visibility="collapsed")

    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip():
            result, scores = predict_sentiment(user_input, pipe)
            if result:
                st.success(f"**Top Sentiment:** {result['label']} — **Confidence:** {result['score']:.2f}")
                st.progress(int(result["score"] * 100))
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    plot_bar_chart(scores)
                with col2:
                    plot_pie_chart(scores)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

    st.markdown("<hr><center><small>Made with ❤️ using Streamlit & Hugging Face</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
