import sqlite3

conn = sqlite3.connect("database/complaints.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM complaints")

rows = cursor.fetchall()

print("\nSaved Complaints:\n")

for row in rows:
    print("ID:", row[0])
    print("Original:", row[1])
    print("Cleaned:", row[2])
    print("Category:", row[3])
    print("Priority:", row[4])
    print("-" * 50)

conn.close()