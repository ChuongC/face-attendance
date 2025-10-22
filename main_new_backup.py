#!/usr/bin/env python3
"""
Optimized Face Attendance main.py
- Integrated Liveness Detection (Blink/LBP/Motion/ONNX)
- Async frame capture
- Parallel embeddings
- HRM online/offline detection
- Safe logging & cleanup
- Background workers for Django + HRM
"""

import os, sys, cv2, time, faiss, logging, threading, requests, argparse, django
import numpy as np
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.utils import timezone

# ---------------- Django setup ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DJANGO_PROJECT_PATH = os.path.join(CURRENT_DIR, "django_admin")
sys.path.append(DJANGO_PROJECT_PATH)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_admin.settings")
django.setup()

# ---------------- Django models & utils ----------------
from employees.models import FaceDataset
from admin_panel.models import Employee, AttendanceRecord
from admin_panel.sync_utils import log_attendance_to_django
from attendance_manager.offline_sync import enqueue_attendance, start_background_sync
from attendance_manager.push_to_hrm import enqueue_hrm
from attendance_manager.attendance_logger import AttendanceLogger
from utils.helpers import draw_fancy_bbox
from sync_faces import sync_faces
import onnxruntime as ort
ort.set_default_logger_severity(3)
from models import SCRFD, ArcFaceONNX

# ---------------- Liveness Ensemble ----------------
from models.liveness import LivenessEnsemble  

# ============================================================
# CONFIG
# ============================================================
HRM_ENDPOINT = "http://127.0.0.1:9000/hrm/webhook"
EMBED_WORKERS = 4
HRM_STATUS_INTERVAL = 10
SIM_THRESH = 0.6
LIVENESS_ONNX_PATH = "./weights/liveness_model.onnx"

# ============================================================
# LOGGING
# ============================================================
def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# HRM STATUS CHECK
# ============================================================
_hrm_online_cache = False
_last_hrm_check = 0

def is_hrm_online():
    global _hrm_online_cache, _last_hrm_check
    now = time.time()
    if now - _last_hrm_check < HRM_STATUS_INTERVAL:
        return _hrm_online_cache
    try:
        r = requests.head(HRM_ENDPOINT, timeout=1)
        _hrm_online_cache = (r.status_code < 400)
        msg = "🟢 HRM ONLINE" if _hrm_online_cache else "🔴 HRM OFFLINE"
        logging.info(f"[HRM STATUS] {msg} (code={r.status_code})")
    except Exception as e:
        _hrm_online_cache = False
        logging.info(f"[HRM STATUS] 🔴 HRM OFFLINE ({e})")
    _last_hrm_check = now
    return _hrm_online_cache

# ============================================================
# FRAME CAPTURE
# ============================================================
class FrameGrabber:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.is_camera = isinstance(src, int) or (isinstance(src, str) and src.isdigit())
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_delay = 1.0 / self.fps
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = f
            # 🔹 Chỉ sleep khi là video file
            if not self.is_camera:
                time.sleep(self.frame_delay)
    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def release(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass

# ============================================================
# FAISS INDEX
# ============================================================
def build_faiss(detector, recognizer, faces_dir, conf_thresh, input_size):
    targets, names = [], []
    for fn in os.listdir(faces_dir):
        if not fn.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(faces_dir, fn)
        img = cv2.imread(path)
        if img is None: continue
        bboxes, kpss = detector.detect(img, input_size=(input_size, input_size),
                                       thresh=conf_thresh, max_num=1)
        if not len(kpss):
            logging.warning(f"No face found in {fn}")
            continue
        emb = recognizer(img, kpss[0]).astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        targets.append(emb)
        names.append(os.path.splitext(fn)[0])
    if not targets:
        raise RuntimeError("No face embeddings found.")
    X = np.vstack(targets).astype(np.float32)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    logging.info(f"[FAISS] Added {len(names)} faces")
    return index, names

# ============================================================
# BACKGROUND WORKERS
# ============================================================
hrm_queue, django_queue = Queue(), Queue()

def hrm_worker():
    while True:
        rec = hrm_queue.get()
        if rec is None: break
        try:
            enqueue_hrm(rec)
            logging.info(f"[HRM ✅] {rec['employee_id']} pushed")
        except Exception as e:
            logging.warning(f"[HRM ❌] {rec['employee_id']} failed ({e}) → saved offline")
            enqueue_attendance(employee_id=rec['employee_id'],
                               similarity=rec.get("similarity", 1.0),
                               source="camera_1")
        hrm_queue.task_done()

def django_worker():
    while True:
        rec = django_queue.get()
        if rec is None: break
        try:
            log_attendance_to_django(rec["employee_id"], similarity=rec.get("similarity", 1.0))
            logging.info(f"[DJANGO ✅] {rec['employee_id']} logged")
        except Exception as e:
            logging.warning(f"[DJANGO ❌] {rec['employee_id']} failed ({e})")
        django_queue.task_done()

threading.Thread(target=hrm_worker, daemon=True).start()
threading.Thread(target=django_worker, daemon=True).start()

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weight", default="./weights/det_10g.onnx")
    parser.add_argument("--rec-weight", default="./weights/w600k_r50.onnx")
    parser.add_argument("--faces-dir", default="./faces")
    parser.add_argument("--source", default="0")
    parser.add_argument("--confidence-thresh", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    start_background_sync()
    sync_faces()

    logging.info("🚀 Initializing models...")
    detector = SCRFD(args.det_weight)
    recognizer = ArcFaceONNX(args.rec_weight)
    liveness = LivenessEnsemble(onnx_path=LIVENESS_ONNX_PATH)

    faiss_index, employee_names = build_faiss(detector, recognizer, args.faces_dir,
                                              args.confidence_thresh, args.input_size)
    grabber = FrameGrabber(int(args.source) if args.source.isdigit() else args.source)
    logger = AttendanceLogger("database/attendance.db")
    executor = ThreadPoolExecutor(max_workers=EMBED_WORKERS)

    last_checkin = {}
    logging.info("✅ System ready. Press 'q' to quit.")

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue

            bboxes, kpss = detector.detect(frame, input_size=(args.input_size, args.input_size),
                                           thresh=args.confidence_thresh, max_num=10)
            if not len(kpss):
                cv2.imshow("Face Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            futures = [executor.submit(lambda f, k: recognizer(f, k).astype(np.float32), frame, kps) for kps in kpss]
            embeddings = [f.result() / (np.linalg.norm(f.result()) + 1e-9) for f in as_completed(futures)]
            X = np.vstack(embeddings)
            faiss.normalize_L2(X)
            D, I = faiss_index.search(X, 1)

            for idx, (sim, i) in enumerate(zip(D[:, 0], I[:, 0])):
                name = employee_names[i] if sim > SIM_THRESH else "Unknown"
                bbox = bboxes[idx]

                # Liveness detection
                x1, y1, x2, y2 = map(int, bbox[:4])
                face_crop = frame[y1:y2, x1:x2]
                if face_crop is None or face_crop.size == 0:
                    logging.warning(f"[CROP ❌] Bỏ qua khuôn mặt {name} vì crop rỗng hoặc sai kích thước.")
                    continue
                live_decision, details = liveness.update(track_id=name, face_crop_bgr=face_crop)
                live_ok = live_decision

                color = (0, 255, 0) if (name != "Unknown" and live_ok) else (0, 0, 255)
                label = f"{name} ({sim:.2f}) {'LIVE' if live_ok else 'FAKE'}"
                draw_fancy_bbox(frame, bbox, name=label, similarity=sim, color=color)

                if name == "Unknown" or not live_ok:
                    if not live_ok and name != "Unknown":
                        logging.warning(f"[LIVENESS ❌] {name} rejected (fake)")
                    continue

                if live_ok:
                    logging.info(f"[LIVENESS ✅] {name} confirmed real")

                now = time.time()
                if name not in last_checkin or now - last_checkin[name] > 10:
                    last_checkin[name] = now
                    record = {"employee_id": name, "timestamp": timezone.now().isoformat(),
                              "similarity": sim, "source": "camera_1"}
                    enqueue_attendance(employee_id=name, similarity=sim, source="camera_1")

                    if is_hrm_online():
                        hrm_queue.put(record)
                        logging.info(f"[ONLINE PUSH] {name} → HRM")
                    else:
                        logging.info(f"[OFFLINE MODE] {name} stored locally")

                    django_queue.put(record)
                else:
                    print(f"[SKIP] {name} vừa chấm công gần đây, bỏ qua.")

            cv2.imshow("Face Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logging.info("🛑 Interrupted by user.")
    finally:
        logging.info("Cleaning up...")
        grabber.release()
        cv2.destroyAllWindows()
        logger.close()
        hrm_queue.put(None)
        django_queue.put(None)
        executor.shutdown(wait=False)
        logging.info("✅ Shutdown complete.")

if __name__ == "__main__":
    main()


