

import sqlite3
import uuid
import time
import requests
import os
import threading
import logging
from datetime import datetime, timedelta
from django.utils.timezone import now, make_aware, is_aware
import pytz

# -------------------- Cấu hình -------------------- #
DB_PATH = os.path.join(os.path.dirname(__file__), "../database/attendance_queue.db")
SYNC_INTERVAL = 10  # giây
SERVER_ENDPOINT = "http://localhost:8000/api/log_attendance/"

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

def determine_status(employee_id, timestamp):
    """Xác định Check-in/Check-out dựa theo lần chấm gần nhất."""
    STATE_FILE = "database/last_checkins.txt"
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    last = {}

    # Đọc lần chấm công trước
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            for line in f:
                eid, ts = line.strip().split(",")
                try:
                    last[eid] = datetime.fromisoformat(ts)
                except Exception:
                    continue

    now_time = datetime.fromisoformat(timestamp)
    if not is_aware(now_time):
        now_time = make_aware(now_time, pytz.UTC)

    status = "Check-in"

    if employee_id in last:
        last_time = last[employee_id]
        if not is_aware(last_time):
            last_time = make_aware(last_time, pytz.UTC)

        delta = now_time - last_time
        if delta < timedelta(minutes=5):
            return None  # tránh spam
        elif delta < timedelta(hours=2):
            status = "Check-out"

    # cập nhật lại file
    last[employee_id] = now_time
    with open(STATE_FILE, "w") as f:
        for k, v in last.items():
            f.write(f"{k},{v.isoformat()}\n")

    return status

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
                # --- FIX: chuyển similarity từ bytes về float nếu cần ---
                if isinstance(sim, bytes):
                    import struct
                    try:
                        sim = struct.unpack('f', sim)[0]  # đọc float 4 bytes
                    except Exception:
                        sim = float.fromhex(sim.hex()) if hasattr(sim, 'hex') else 0.0

                # Ép kiểu an toàn các giá trị khác
                def safe_decode(x):
                    if isinstance(x, bytes):
                        return x.decode(errors='ignore')
                    return str(x) if not isinstance(x, (int, float)) else x

                emp_id = safe_decode(emp_id)
                source = safe_decode(source)
                ts = safe_decode(ts)

                status = determine_status(emp_id, ts)
                if status is None:
                    logging.info(f"[SKIP] Duplicate for {emp_id}, ignored.")
                    continue

                payload = {
                    "employee_id": emp_id,
                    "similarity": float(sim),  # đảm bảo là float
                    "timestamp": ts,
                    "source": source
                }

                print(f"[DEBUG] Payload to send -> {payload}")

                resp = requests.post(SERVER_ENDPOINT, json=payload, timeout=5)
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
