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


# Split the data BEFORE TF-IDF
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


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit TF-IDF ONLY on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)


# Train category model
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train_tfidf, y_category_train)


# Train priority model
priority_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
priority_model.fit(X_train_tfidf, y_priority_train)


# Predictions
category_predictions = category_model.predict(X_test_tfidf)
priority_predictions = priority_model.predict(X_test_tfidf)


# Calculate accuracy
category_accuracy = accuracy_score(
    y_category_test,
    category_predictions
)

priority_accuracy = accuracy_score(
    y_priority_test,
    priority_predictions
)


# Create model directory
os.makedirs("model", exist_ok=True)


# Save models
joblib.dump(
    category_model,
    "model/category_model.pkl"
)

joblib.dump(
    priority_model,
    "model/priority_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)


# Display results
print("Training completed successfully!")
print(f"Category model accuracy: {category_accuracy * 100:.2f}%")
print(f"Priority model accuracy: {priority_accuracy * 100:.2f}%")
print("Category model saved.")
print("Priority model saved.")
print("Vectorizer saved.")