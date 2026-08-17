import pandas as pd
import re


df = pd.read_csv("data/complaints.csv")

print("Original dataset:")
print(df.head())


def clean_text(text):
    text = text.lower()                     # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove numbers/special characters
    text = re.sub(r"\s+", " ", text)        # Remove extra spaces
    return text.strip()


df["complaint"] = df["complaint"].apply(clean_text)


print("\nCleaned dataset:")
print(df.head())


df.to_csv("data/processed.csv", index=False)

print("\nProcessed dataset saved successfully!")