# import sqlite3
# import uuid
# import time
# import requests
# import os
# from datetime import datetime

# DB_PATH = os.path.join(os.path.dirname(__file__), "../database/attendance_queue.db")
# SYNC_INTERVAL = 10  # giây
# SERVER_ENDPOINT = "http://localhost:8000/api/log_attendance"

# # -------------------- Khởi tạo bảng -------------------- #
# def init_db():
#     os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # ✅ tạo thư mục nếu chưa có
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS queue (
#         id TEXT PRIMARY KEY,
#         employee_id TEXT,
#         similarity REAL,
#         source TEXT,
#         timestamp TEXT,
#         synced INTEGER DEFAULT 0
#     )
#     ''')
#     conn.commit()
#     conn.close()
#     print(f"[INIT] SQLite queue DB ready at {os.path.abspath(DB_PATH)}")

# # ✅ Gọi ngay khi import
# init_db()

# # -------------------- Thêm record vào queue -------------------- #
# def enqueue_attendance(employee_id, similarity=1.0, source="camera"):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     record_id = str(uuid.uuid4())
#     timestamp = datetime.now().isoformat()
#     c.execute('''
#         INSERT INTO queue (id, employee_id, similarity, source, timestamp, synced)
#         VALUES (?, ?, ?, ?, ?, 0)
#     ''', (record_id, employee_id, similarity, source, timestamp))
#     conn.commit()
#     conn.close()
#     print(f"[QUEUE] Enqueued attendance {employee_id} ({record_id})")

# # -------------------- Worker sync -------------------- #
# def sync_worker():
#     while True:
#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()
#         c.execute("SELECT * FROM queue WHERE synced=0")
#         rows = c.fetchall()
#         for row in rows:
#             record_id, emp_id, sim, source, ts, synced = row
#             try:
#                 resp = requests.post(SERVER_ENDPOINT, json={
#                     "employee_id": emp_id,
#                     "similarity": sim,
#                     "timestamp": ts,
#                     "source": source
#                 }, timeout=5)
#                 if resp.status_code == 200:
#                     c.execute("UPDATE queue SET synced=1 WHERE id=?", (record_id,))
#                     print(f"[SYNC ✅] {emp_id} ({record_id})")
#                 else:
#                     print(f"[SYNC ⚠] Server response: {resp.status_code}")
#             except Exception as e:
#                 print(f"[SYNC ❌] Failed for {emp_id}: {e}")
#         conn.commit()
#         conn.close()
#         time.sleep(SYNC_INTERVAL)

# # -------------------- Test chạy nhanh -------------------- #
# if __name__ == "__main__":
#     init_db()
#     enqueue_attendance("E001")  # test thêm
#     # sync_worker()  # test worker


import sqlite3
import uuid
import time
import requests
import os
import threading
import logging
from datetime import datetime
from django.utils.timezone import now
# -------------------- Cấu hình -------------------- #
DB_PATH = os.path.join(os.path.dirname(__file__), "../database/attendance_queue.db")
SYNC_INTERVAL = 10  # giây
SERVER_ENDPOINT = "http://localhost:8000/api/log_attendance"

# -------------------- Thiết lập logging -------------------- #
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/sync.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -------------------- Khởi tạo DB -------------------- #
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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

# Gọi khi import
init_db()

# -------------------- Thêm record vào queue -------------------- #
def enqueue_attendance(employee_id, similarity=1.0, source="camera"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    record_id = str(uuid.uuid4())
    #timestamp = datetime.now().isoformat()
    timestamp = now().isoformat()
    c.execute('''
        INSERT INTO queue (id, employee_id, similarity, source, timestamp, synced)
        VALUES (?, ?, ?, ?, ?, 0)
    ''', (record_id, employee_id, similarity, source, timestamp))
    conn.commit()
    conn.close()
    msg = f"[QUEUE] Enqueued attendance {employee_id} ({record_id})"
    print(msg)
    logging.info(msg)

# -------------------- Worker đồng bộ nền -------------------- #
def sync_worker():
    print("[SYNC] Background worker started...")
    logging.info("Background sync worker started.")
    while True:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM queue WHERE synced=0")
        rows = c.fetchall()

        if not rows:
            logging.debug("No unsynced records found.")
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
                    msg = f"[SYNC ✅] Synced {emp_id} ({record_id})"
                    print(msg)
                    logging.info(msg)
                else:
                    msg = f"[SYNC ⚠] Server error {resp.status_code} for {emp_id}"
                    print(msg)
                    logging.warning(msg)
            except Exception as e:
                msg = f"[SYNC ❌] Failed for {emp_id}: {e}"
                print(msg)
                logging.error(msg)

        conn.commit()
        conn.close()
        time.sleep(SYNC_INTERVAL)

# -------------------- Khởi động thread nền -------------------- #
def start_background_sync():
    """Khởi động thread đồng bộ nền (daemon)."""
    thread = threading.Thread(target=sync_worker, daemon=True)
    thread.start()
    print("[SYNC] Background sync thread launched.")
    logging.info("Background sync thread launched.")
    return thread

# -------------------- Test độc lập -------------------- #
if __name__ == "__main__":
    init_db()
    enqueue_attendance("E001", source="test")
    start_background_sync()
    while True:
        time.sleep(1)
