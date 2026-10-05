import sqlite3
import os

DB_PATH = "database/complaints.db"

def create_database():
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
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


def save_complaint(original, cleaned, category, priority):
    create_database()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO complaints
    (original_complaint, cleaned_complaint, category, priority)
    VALUES (?, ?, ?, ?)
    """, (original, cleaned, category, priority))

    conn.commit()

    complaint_id = cursor.lastrowid

    conn.close()

    return complaint_id


if __name__ == "__main__":
    create_database()
    print("Database created successfully!")