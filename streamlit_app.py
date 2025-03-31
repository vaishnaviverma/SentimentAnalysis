import streamlit as st
from tensorflow.keras.models import load_model
from utils import encode_review

# Load model
model = load_model("sentiment_model.h5")

# Page config
st.set_page_config(page_title="Movie Review Sentiment", layout="centered")

st.title("Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below, and the model will tell you if it's positive or negative.")

# Text input
user_input = st.text_area("Your Review", height=150)

# Predict
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review first!")
    else:
        encoded = encode_review(user_input)
        prediction = model.predict(encoded)[0][0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        confidence = round(float(prediction), 3)

        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** `{confidence}`")

"""To run:
streamlit run streamlit_app.py

"""

