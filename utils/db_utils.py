import sqlite3
from typing import List, Tuple

DB_PATH = "./database/attendance.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_all_attendance() -> List[Tuple[int, str, str]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
