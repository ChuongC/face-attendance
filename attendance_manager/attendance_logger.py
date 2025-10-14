import sqlite3
from datetime import datetime
import os

class AttendanceLogger:
    def __init__(self, db_path="database/attendance.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def log(self, name: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO attendance (name, timestamp) VALUES (?, ?)", 
            (name, now)
        )
        self.conn.commit()
        print(f"[LOG] {name} checked in at {now}")

    def close(self):
        self.conn.close()

