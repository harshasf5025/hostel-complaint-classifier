import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


# Load processed dataset
data = pd.read_csv("data/processed.csv")

# Input and targets
X = data["complaint"]
y_category = data["category"]
y_priority = data["priority"]

# Convert complaint text into TF-IDF features
vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_category_train, y_category_test, y_priority_train, y_priority_test = train_test_split(
    X_tfidf,
    y_category,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

# Train category model
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train, y_category_train)

# Train priority model
priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_train, y_priority_train)

# Evaluate each classifier on the held-out test set.
category_accuracy = accuracy_score(y_category_test, category_model.predict(X_test))
priority_accuracy = accuracy_score(y_priority_test, priority_model.predict(X_test))

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save models and vectorizer
joblib.dump(category_model, "model/category_model.pkl")
joblib.dump(priority_model, "model/priority_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Training completed successfully!")
print(f"Category model accuracy: {category_accuracy:.2%}")
print(f"Priority model accuracy: {priority_accuracy:.2%}")
print("Category model saved.")
print("Priority model saved.")
print("Vectorizer saved.")
