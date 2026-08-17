import pandas as pd

# Load dataset
df = pd.read_csv("data/complaints.csv")

# Basic information
print("========== DATASET INFORMATION ==========")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

# Display first 5 complaints
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Category distribution
print("\n========== CATEGORY DISTRIBUTION ==========")
print(df["category"].value_counts())

# Priority distribution
print("\n========== PRIORITY DISTRIBUTION ==========")
print(df["priority"].value_counts())