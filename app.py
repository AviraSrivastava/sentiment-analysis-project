
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("sentiment_model.pkl")

# Setup page config
st.set_page_config(page_title="🎬 Movie Sentiment App", layout="centered")


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []


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

# Prediction History (inside expander)
with st.expander("📜 View Prediction History"):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        # Pie Chart using Plotly
        st.subheader("📊 Sentiment Distribution")
        pie = px.pie(hist_df, names="Prediction", title="Summary of Sentiments", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(pie, use_container_width=True)

        # Download / Clear
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV", hist_df.to_csv(index=False), "prediction_history.csv", "text/csv")
        with col2:
            if st.button("🧹 Clear History"):
                st.session_state.history.clear()
                st.experimental_rerun()
    else:
        st.info("No predictions yet.")