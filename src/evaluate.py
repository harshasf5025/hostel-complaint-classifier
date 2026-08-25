import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load processed dataset
data = pd.read_csv("data/processed.csv")


# Load trained models
category_model = joblib.load("model/category_model.pkl")
priority_model = joblib.load("model/priority_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# Input and target data
X = data["complaint"]
y_category = data["category"]
y_priority = data["priority"]


# Use the SAME train/test split used during training
(
    X_train,
    X_test,
    y_category_train,
    y_category_test,
    y_priority_train,
    y_priority_test
) = train_test_split(
    X,
    y_category,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)


# Convert ONLY the test complaints using the saved vectorizer
X_test_tfidf = vectorizer.transform(X_test)


# Make predictions
category_predictions = category_model.predict(X_test_tfidf)
priority_predictions = priority_model.predict(X_test_tfidf)


# ==============================
# CATEGORY MODEL
# ==============================

print("\n========== CATEGORY MODEL ==========")

print(
    f"Accuracy: "
    f"{accuracy_score(y_category_test, category_predictions) * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_category_test,
        category_predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_category_test,
        category_predictions
    )
)


# ==============================
# PRIORITY MODEL
# ==============================

print("\n========== PRIORITY MODEL ==========")

print(
    f"Accuracy: "
    f"{accuracy_score(y_priority_test, priority_predictions) * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_priority_test,
        priority_predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_priority_test,
        priority_predictions
    )
)