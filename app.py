
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("sentiment_model.pkl")

# Setup page config
st.set_page_config(page_title="🎬 Movie Sentiment App", layout="centered")

# Title & Description
st.markdown("<h1 style='text-align: center;'>🎥 Movie Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Instantly predict the sentiment of your movie reviews!</p>", unsafe_allow_html=True)
st.divider()

# Input box
review = st.text_area("📝 Write a review:", placeholder="Type your movie review here...", height=150)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please write a review before predicting.")
    else:
        prediction = model.predict([review])[0]
        confidence = round(max(model.predict_proba([review])[0]) * 100, 2)

        # Display result
        st.subheader("🎯 Sentiment:")
        if prediction == "positive":
            st.success(f"👍 POSITIVE ({confidence}%)")
        else:
            st.error(f"👎 NEGATIVE ({confidence}%)")

        # Store in history
        st.session_state.history.append({
            "Review": review,
            "Prediction": prediction,
            "Confidence": confidence
        })

# Divider
st.divider()
