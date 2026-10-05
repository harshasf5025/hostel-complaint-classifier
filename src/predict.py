import re
import joblib
from database import save_complaint

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

vectorizer = joblib.load("model/vectorizer.pkl")
category_model = joblib.load("model/category_model.pkl")
priority_model = joblib.load("model/priority_model.pkl")

complaint = input("Enter your complaint: ")

cleaned_complaint = clean_text(complaint)

complaint_tfidf = vectorizer.transform([cleaned_complaint])

category = category_model.predict(complaint_tfidf)[0]
priority = priority_model.predict(complaint_tfidf)[0]

complaint_id = save_complaint(
    complaint,
    cleaned_complaint,
    category,
    priority
)

print("\nOriginal complaint:")
print(complaint)

print("\nCleaned complaint:")
print(cleaned_complaint)

print("\nPrediction:")
print("Category:", category)
print("Priority:", priority)

print("\nComplaint saved to database!")
print("Complaint ID:", complaint_id)