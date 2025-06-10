import sqlite3
import pandas as pd
import time
import os

DB_PATH = 'predictions.db'

def read_and_print_db():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()

        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen for better output
        print("🔄 Latest Predictions from DB:\n")
        print(df)
    except Exception as e:
        print("Error reading DB:", e)

def main():
    print("Starting DB polling service (every 30 seconds)...\nPress Ctrl+C to stop.\n")
    while True:
        read_and_print_db()
        time.sleep(30)

if __name__ == "__main__":
    main()
