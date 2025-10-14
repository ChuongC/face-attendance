import sqlite3
import uuid
import time
import requests
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/attendance_queue.db")
SYNC_INTERVAL = 10  # giây
SERVER_ENDPOINT = "http://localhost:8000/api/log_attendance"

# -------------------- Khởi tạo bảng -------------------- #
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # ✅ tạo thư mục nếu chưa có
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS queue (
        id TEXT PRIMARY KEY,
        employee_id TEXT,
        similarity REAL,
        source TEXT,
        timestamp TEXT,
        synced INTEGER DEFAULT 0
    )
    ''')
    conn.commit()
    conn.close()
    print(f"[INIT] SQLite queue DB ready at {os.path.abspath(DB_PATH)}")

# ✅ Gọi ngay khi import
init_db()

# -------------------- Thêm record vào queue -------------------- #
def enqueue_attendance(employee_id, similarity=1.0, source="camera"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    record_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute('''
        INSERT INTO queue (id, employee_id, similarity, source, timestamp, synced)
        VALUES (?, ?, ?, ?, ?, 0)
    ''', (record_id, employee_id, similarity, source, timestamp))
    conn.commit()
    conn.close()
    print(f"[QUEUE] Enqueued attendance {employee_id} ({record_id})")

# -------------------- Worker sync -------------------- #
def sync_worker():
    while True:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM queue WHERE synced=0")
        rows = c.fetchall()
        for row in rows:
            record_id, emp_id, sim, source, ts, synced = row
            try:
                resp = requests.post(SERVER_ENDPOINT, json={
                    "employee_id": emp_id,
                    "similarity": sim,
                    "timestamp": ts,
                    "source": source
                }, timeout=5)
                if resp.status_code == 200:
                    c.execute("UPDATE queue SET synced=1 WHERE id=?", (record_id,))
                    print(f"[SYNC ✅] {emp_id} ({record_id})")
                else:
                    print(f"[SYNC ⚠] Server response: {resp.status_code}")
            except Exception as e:
                print(f"[SYNC ❌] Failed for {emp_id}: {e}")
        conn.commit()
        conn.close()
        time.sleep(SYNC_INTERVAL)

# -------------------- Test chạy nhanh -------------------- #
if __name__ == "__main__":
    init_db()
    enqueue_attendance("E001")  # test thêm
    # sync_worker()  # test worker
