import streamlit as st
import sys
import os
import re
import joblib

# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from database import save_complaint


# Page settings
st.set_page_config(
    page_title="Hostel Complaint Classifier",
    page_icon="🏠",
    layout="centered"
)


# Load trained models
vectorizer = joblib.load("model/vectorizer.pkl")
category_model = joblib.load("model/category_model.pkl")
priority_model = joblib.load("model/priority_model.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Title
st.title("🏠 Hostel Complaint Classifier")
st.write("Enter your hostel complaint below and the system will predict its category and priority.")


# Complaint input
complaint = st.text_area(
    "Enter your complaint:",
    placeholder="Example: WiFi is not working in my room...",
    height=150
)


# Predict button
if st.button("Predict Complaint"):

    if complaint.strip() == "":
        st.warning("Please enter a complaint.")

    else:
        # Clean complaint
        cleaned_complaint = clean_text(complaint)

        # Convert to TF-IDF
        complaint_tfidf = vectorizer.transform([cleaned_complaint])

        # Predictions
        category = category_model.predict(complaint_tfidf)[0]
        priority = priority_model.predict(complaint_tfidf)[0]

        # Save to database
        complaint_id = save_complaint(
            complaint,
            cleaned_complaint,
            category,
            priority
        )

        # Display results
        st.success("Complaint processed successfully!")

        st.subheader("Prediction Result")

        st.write("**Category:**", category)
        st.write("**Priority:**", priority)

        st.subheader("Processed Complaint")
        st.write(cleaned_complaint)

        st.info(f"Complaint saved to database with ID: {complaint_id}")