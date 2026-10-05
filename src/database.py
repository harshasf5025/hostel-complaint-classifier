import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/complaints.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_complaint TEXT NOT NULL,
    cleaned_complaint TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL
)
""")

conn.commit()
conn.close()

print("Database created successfully!")