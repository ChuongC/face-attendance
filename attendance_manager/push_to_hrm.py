import requests
import time
import logging
from queue import Queue
from threading import Thread

HRM_ENDPOINT = "http://localhost:9000/hrm/webhook"
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# Thread-safe queue
attendance_queue = Queue()

def push_attendance(record):
    """
    record = {
        "employee_id": str,
        "timestamp": str (ISO),
        "similarity": float,
        "source": str
    }
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            r = requests.post(HRM_ENDPOINT, json=record, timeout=5)
            if r.status_code == 200:
                logging.info(f"[HRM] Attendance pushed: {record}")
                return True
            else:
                logging.warning(f"[HRM] Failed {r.status_code}: {r.text}")
        except Exception as e:
            logging.warning(f"[HRM] Exception: {e}")
        retries += 1
        time.sleep(RETRY_DELAY)
    logging.error(f"[HRM] Max retries reached, failed to push: {record}")
    return False

def enqueue_hrm(record):
    attendance_queue.put(record)

def hrm_worker():
    while True:
        record = attendance_queue.get()
        if record is None:
            break
        push_attendance(record)
        attendance_queue.task_done()

# Start background thread
worker_thread = Thread(target=hrm_worker, daemon=True)
worker_thread.start()
