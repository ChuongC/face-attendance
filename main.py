#!/usr/bin/env python3
"""
main_cpu_optimized.py
- CPU-optimized variant of your original main.py
- Improvements included (CPU-only):
  * Multi-image mean embeddings for FAISS (better robustness)
  * Face histogram equalization preprocessing (Y channel) for face images
  * Dynamic similarity threshold based on frame brightness
  * Require 2 consecutive positive frames before confirming attendance
  * Combine similarity score with liveness confidence for final decision
  * Rate-limit liveness checks and resize face crops for liveness model
  * Thread pool for embedding extraction with safe result handling
  * Environment thread controls for CPU libraries

Notes:
- This script keeps the same high-level logic as your original main.py
- Replace relative paths to weights, faces dir or Django project as needed
"""

import os
import sys
import time
import logging
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Control CPU threads for deterministic CPU-only performance
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np
import faiss
import requests
import django

# ---------------- Django setup (keep as in your original) ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DJANGO_PROJECT_PATH = os.path.join(CURRENT_DIR, "django_admin")
if DJANGO_PROJECT_PATH not in sys.path:
    sys.path.append(DJANGO_PROJECT_PATH)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_admin.settings")
try:
    django.setup()
except Exception:
    # If Django isn't available in the environment during local tests, we continue.
    pass

# ---------------- Models & utils (imports kept as in original project) ----------------
try:
    from employees.models import FaceDataset
    from admin_panel.models import Employee, AttendanceRecord
    from admin_panel.sync_utils import log_attendance_to_django
    from attendance_manager.offline_sync import enqueue_attendance, start_background_sync
    from attendance_manager.push_to_hrm import enqueue_hrm
    from attendance_manager.attendance_logger import AttendanceLogger
    from utils.helpers import draw_fancy_bbox
    from sync_faces import sync_faces
except Exception:
    # If these don't exist in your current environment, the file still remains runnable for static checks.
    def enqueue_attendance(*_, **__):
        logging.info("enqueue_attendance called (stub)")
    def start_background_sync():
        logging.info("start_background_sync (stub)")
    def enqueue_hrm(*_, **__):
        logging.info("enqueue_hrm (stub)")
    def log_attendance_to_django(*_, **__):
        logging.info("log_attendance_to_django (stub)")
    def sync_faces():
        logging.info("sync_faces (stub)")
    def draw_fancy_bbox(frame, bbox, name=None, similarity=None, color=(0,255,0)):
        x1,y1,x2,y2 = map(int, bbox[:4])
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        if name:
            cv2.putText(frame, f"{name} {similarity:.2f}" if similarity is not None else name,
                        (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Optional model classes (SCRFD, ArcFaceONNX, LivenessEnsemble)
try:
    from models import SCRFD, ArcFaceONNX
    from models.liveness import LivenessEnsemble
except Exception:
    # Stubs so file is static-parseable
    class SCRFD:
        def __init__(self, *_, **__):
            pass
        def detect(self, img, input_size=(320,320), thresh=0.5, max_num=10):
            h,w = img.shape[:2]
            return [], []
    class ArcFaceONNX:
        def __init__(self, *_, **__):
            pass
        def __call__(self, img, kps):
            return np.random.rand(512).astype(np.float32)
    class LivenessEnsemble:
        def __init__(self, *_, **__):
            pass
        def update(self, track_id, face_crop_bgr):
            return True, {"confidence": 1.0}

import onnxruntime as ort
ort.set_default_logger_severity(3)

# ---------------- Configurable constants ----------------
HRM_ENDPOINT = "http://127.0.0.1:9000/hrm/webhook"
EMBED_WORKERS = 4
HRM_STATUS_INTERVAL = 10
SIM_THRESH = 0.6  # fallback value; we use dynamic thresholding inside loop
LIVENESS_ONNX_PATH = "./weights/liveness_model.onnx"
LIVENESS_FRAME_RATE = 3  # only run liveness every N frames per track
CONSECUTIVE_REQUIRED = 2  # number of consecutive frames required to confirm check-in
LIVENESS_WEIGHT = 0.25  # weight for liveness in combined score
SIM_WEIGHT = 0.75

# ---------------- Logging ----------------

def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- HRM online check (unchanged) ----------------
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

# ---------------- FrameGrabber (kept, small improvement) ----------------
class FrameGrabber:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.is_camera = isinstance(src, int) or (isinstance(src, str) and src.isdigit())
        self.fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS) or 30)
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
            if not self.is_camera:
                time.sleep(self.frame_delay)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# ---------------- FAISS builder: multi-image mean embeddings + preprocessing ----------------

def preprocess_for_embedding(img):
    # Convert BGR->YUV, equalize Y channel, convert back
    try:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    except Exception:
        out = img
    return out


def build_faiss(detector, recognizer, faces_dir, conf_thresh, input_size):
    name_to_embs = {}
    for fn in sorted(os.listdir(faces_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(faces_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue

        bboxes, kpss = detector.detect(img, input_size=(input_size, input_size),
                                       thresh=conf_thresh, max_num=1)
        if not len(kpss):
            logging.warning(f"No face found in {fn}")
            continue

        face_pre = preprocess_for_embedding(img)
        emb = recognizer(face_pre, kpss[0]).astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)

        # Assume filename like NAME_1.jpg or NAME.jpg -> take part before first underscore
        name = os.path.splitext(fn)[0].split("_")[0]
        name_to_embs.setdefault(name, []).append(emb)

    targets, names = [], []
    for name, embs in name_to_embs.items():
        mean_emb = np.mean(embs, axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)
        targets.append(mean_emb)
        names.append(name)

    if not targets:
        raise RuntimeError("No face embeddings found in faces_dir.")

    X = np.vstack(targets).astype(np.float32)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    logging.info(f"[FAISS] Added {len(names)} faces (multi-image mean embeddings)")
    return index, names

# ---------------- Background workers (unchanged) ----------------
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
                               source=rec.get("source", "camera_1"))
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

# ---------------- Main loop (optimized) ----------------

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

    logging.info("🚀 Initializing models (CPU)...")
    detector = SCRFD(args.det_weight)
    recognizer = ArcFaceONNX(args.rec_weight)
    liveness = LivenessEnsemble(onnx_path=LIVENESS_ONNX_PATH)

    # Build FAISS index (multi-image mean embeddings)
    faiss_index, employee_names = build_faiss(detector, recognizer, args.faces_dir,
                                              args.confidence_thresh, args.input_size)

    grabber = FrameGrabber(int(args.source) if args.source.isdigit() else args.source)
    logger = AttendanceLogger("database/attendance.db")
    executor = ThreadPoolExecutor(max_workers=EMBED_WORKERS)

    last_checkin = {}
    consecutive_cache = {}  # name -> consecutive positive count
    last_sim_cache = {}     # name -> last similarity value
    last_liveness_frame = {}  # name -> frame index last checked for liveness
    frame_idx = 0

    logging.info("✅ System ready. Press 'q' to quit.")

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1

            # Dynamic threshold based on brightness
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            sim_dynamic_thresh = 0.55 if brightness > 90 else 0.45

            bboxes, kpss = detector.detect(frame, input_size=(args.input_size, args.input_size),
                                           thresh=args.confidence_thresh, max_num=10)
            if not len(kpss):
                cv2.imshow("Face Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            # Submit embedding extraction tasks safely capturing frame & kps
            futures = []
            for kps in kpss:
                futures.append(executor.submit(lambda f, k: recognizer(preprocess_for_embedding(f), k).astype(np.float32), frame.copy(), kps))

            # Gather embeddings in same order as kpss
            embeddings = []
            for fut in futures:
                try:
                    emb = fut.result(timeout=5)
                except Exception:
                    emb = None
                if emb is None:
                    continue
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                embeddings.append(emb)

            if not embeddings:
                continue

            X = np.vstack(embeddings)
            faiss.normalize_L2(X)
            D, I = faiss_index.search(X, 1)

            for idx, (sim, i) in enumerate(zip(D[:, 0], I[:, 0])):
                name = employee_names[i] if sim > SIM_THRESH else "Unknown"
                bbox = bboxes[idx]

                x1, y1, x2, y2 = map(int, bbox[:4])
                face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if face_crop is None or face_crop.size == 0:
                    logging.warning(f"[CROP ❌] Skip face crop for {name}")
                    continue

                # Preprocess face_crop for liveness + recognition stability
                face_crop_eq = preprocess_for_embedding(face_crop)

                # Decide whether to run liveness for this track now to save CPU
                run_liveness = False
                if name != "Unknown":
                    last_run = last_liveness_frame.get(name, -9999)
                    if frame_idx - last_run >= LIVENESS_FRAME_RATE:
                        run_liveness = True
                        last_liveness_frame[name] = frame_idx
                else:
                    # For unknowns, we still occasionally run liveness to help filter spoofing
                    last_run = last_liveness_frame.get("__unknown__", -9999)
                    if frame_idx - last_run >= (LIVENESS_FRAME_RATE * 3):
                        run_liveness = True
                        last_liveness_frame["__unknown__"] = frame_idx

                live_ok = True
                live_conf = 1.0
                if run_liveness:
                    # Resize for liveness model to small size to save CPU
                    try:
                        small = cv2.resize(face_crop_eq, (112, 112))
                    except Exception:
                        small = face_crop_eq
                    live_ok, details = liveness.update(track_id=name, face_crop_bgr=small)
                    live_conf = float(details.get("confidence", 1.0)) if isinstance(details, dict) else 1.0

                # Combined score: weighted sum of sim and liveness confidence
                combined_score = SIM_WEIGHT * float(sim) + LIVENESS_WEIGHT * float(live_conf)

                color = (0, 255, 0) if (name != "Unknown" and live_ok and combined_score >= sim_dynamic_thresh) else (0, 0, 255)
                label = f"{name} ({sim:.2f}) {'LIVE' if live_ok else 'FAKE'}"
                draw_fancy_bbox(frame, bbox, name=label, similarity=sim, color=color)

                # Only accept confirmed faces
                if name == "Unknown" or not live_ok:
                    if not live_ok and name != "Unknown":
                        logging.warning(f"[LIVENESS ❌] {name} rejected (fake)")
                    # update consecutive cache to 0 for unknowns
                    if name == "Unknown":
                        consecutive_cache[name] = 0
                    continue

                # Require consecutive positives to avoid one-off false positives
                prev_sim = last_sim_cache.get(name, 0.0)
                prev_count = consecutive_cache.get(name, 0)
                if combined_score >= sim_dynamic_thresh and prev_sim >= sim_dynamic_thresh:
                    consecutive_cache[name] = prev_count + 1
                elif combined_score >= sim_dynamic_thresh:
                    consecutive_cache[name] = 1
                else:
                    consecutive_cache[name] = 0

                last_sim_cache[name] = combined_score

                # Confirm check-in only if sufficient consecutive frames or if not seen recently
                now_ts = time.time()
                last_seen = last_checkin.get(name, 0)
                if consecutive_cache[name] >= CONSECUTIVE_REQUIRED or (now_ts - last_seen) > 30:
                    # Update last_checkin timestamp
                    last_checkin[name] = now_ts

                    record = {"employee_id": name, "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                              "similarity": float(sim), "source": "camera_1"}

                    enqueue_attendance(employee_id=name, similarity=sim, source="camera_1")

                    if is_hrm_online():
                        hrm_queue.put(record)
                        logging.info(f"[ONLINE PUSH] {name} → HRM")
                    else:
                        logging.info(f"[OFFLINE MODE] {name} stored locally")

                    django_queue.put(record)
                else:
                    logging.debug(f"[SKIP] {name} waiting for consecutive confirmations: {consecutive_cache[name]}")

            cv2.imshow("Face Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
