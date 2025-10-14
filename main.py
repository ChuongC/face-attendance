"""
main_liveness_faiss.py
- SCRFD detection (tunable input_size)
- ArcFace embeddings -> FAISS (IndexFlatIP normalized) recognition
- Liveness: LBP texture check + Blink detection via MediaPipe FaceMesh (EAR)
- Attendance logging via AttendanceLogger + offline queue + Django sync

Run:
python main_liveness_faiss.py --source 0
"""

import os
import sys
import cv2
import numpy as np
import argparse
import faiss
import pickle
import logging
import time
import random
import django

import os
import sys

# --- xác định project root tuyệt đối ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DJANGO_PROJECT_PATH = os.path.join(CURRENT_DIR, "django_admin")   # đổi tên nếu khác
sys.path.append(DJANGO_PROJECT_PATH)

# --- thiết lập môi trường Django ---
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_admin.settings")

import django
django.setup()

# === Django models / sync utils ===
from employees.models import FaceDataset
from admin_panel.models import Employee, AttendanceRecord

from admin_panel.sync_utils import log_attendance_to_django
from sync_faces import sync_faces
sync_faces()

# === Local utilities ===
from models import SCRFD, ArcFaceONNX
from utils.helpers import draw_fancy_bbox
from attendance_manager.attendance_logger import AttendanceLogger
from attendance_manager.offline_sync import enqueue_attendance

# === MediaPipe ===
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# ---------------- Argument parsing ---------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx")
    p.add_argument("--rec-weight", type=str, default="./weights/w600k_r50.onnx")
    p.add_argument("--faces-dir", type=str, default="./faces")
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--similarity-thresh", type=float, default=0.45)
    p.add_argument("--confidence-thresh", type=float, default=0.5)
    p.add_argument("--input-size", type=int, default=320)
    p.add_argument("--max-num", type=int, default=5)
    p.add_argument("--use-blink", type=int, default=1)
    p.add_argument("--lbp-thresh", type=float, default=0.01)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()

def setup_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), None),
                        format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Liveness utils ---------------- #
def simple_texture_check(face_crop, std_thresh=0.01):
    if face_crop is None or face_crop.size == 0:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64,64))
    lap = cv2.Laplacian(small, cv2.CV_32F)
    hist = cv2.calcHist([cv2.convertScaleAbs(lap)], [0], None, [64], [0,256])
    hist = hist / (hist.sum() + 1e-9)
    return float(np.std(hist)) > std_thresh

def eye_aspect_ratio(pts):
    if pts.shape[0] < 6:
        return 0.0
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3]) + 1e-9
    return (A + B) / (2.0 * C)

class BlinkState:
    def __init__(self, ear_thresh=0.21, consecutive=2):
        self.ear_thresh = ear_thresh
        self.consec = consecutive
        self.ears = []
        self.last_blinked_time = 0.0

    def update(self, ear):
        self.ears.append(ear)
        if len(self.ears) > self.consec:
            self.ears.pop(0)
        if len(self.ears) == self.consec and all(e < self.ear_thresh for e in self.ears):
            now = time.time()
            if now - self.last_blinked_time > 0.6:
                self.last_blinked_time = now
                self.ears = []
                return True
        return False

# ---------------- MediaPipe FaceMesh ---------------- #
MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

class MediapipeHelper:
    def __init__(self):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not installed")
        self.mp_face = mp.solutions.face_mesh
        self.fm = self.mp_face.FaceMesh(static_image_mode=False,
                                        max_num_faces=5,
                                        refine_landmarks=False,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
    def get_landmarks(self, rgb_frame):
        results = self.fm.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        all_landmarks = []
        for lm in results.multi_face_landmarks:
            coords = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
            all_landmarks.append(coords)
        return all_landmarks

# ---------------- Build FAISS index ---------------- #
def build_targets_and_faiss(detector, recognizer, faces_dir, conf_thresh, input_size):
    targets, names = [], []
    for fn in os.listdir(faces_dir):
        if not fn.lower().endswith((".jpg",".png",".jpeg")):
            continue
        path = os.path.join(faces_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        bboxes, kpss = detector.detect(img, input_size=(input_size,input_size), thresh=conf_thresh, max_num=1)
        if len(kpss) == 0:
            logging.warning(f"No face found for {fn}")
            continue
        emb = recognizer(img, kpss[0]).astype(np.float32)
        emb = emb / (np.linalg.norm(emb)+1e-9)
        targets.append(emb)
        names.append(os.path.splitext(fn)[0])
        logging.info(f"Added {fn} -> {names[-1]}")
    if len(targets) == 0:
        raise RuntimeError("No face embeddings found!")
    emb_np = np.vstack(targets)
    faiss.normalize_L2(emb_np)
    dim = emb_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_np)
    return index, names

# ---------------- Main loop ---------------- #
def main():
    args = parse_args()
    setup_logging(args.log_level)

    detector = SCRFD(args.det_weight)
    recognizer = ArcFaceONNX(args.rec_weight)

    logging.info("Building FAISS index...")
    faiss_index, employee_names = build_targets_and_faiss(detector, recognizer,
                                                          args.faces_dir, args.confidence_thresh,
                                                          args.input_size)
    colors = {n:(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for n in employee_names}

    logger = AttendanceLogger("database/attendance.db")

    mp_helper = MediapipeHelper() if args.use_blink and MP_AVAILABLE else None
    tracker_states = {}
    next_track_id = 0
    TRACK_DIST_THR = 80

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes, kpss = detector.detect(frame, input_size=(args.input_size,args.input_size),
                                           thresh=args.confidence_thresh, max_num=args.max_num)
            mp_landmarks = mp_helper.get_landmarks(rgb) if mp_helper else None
            used_track_ids = set()

            for i,(bbox,kps) in enumerate(zip(bboxes,kpss)):
                x1,y1,x2,y2,_ = bbox.astype(int)
                x1 = max(0,x1); y1=max(0,y1); x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
                centroid = ((x1+x2)//2,(y1+y2)//2)

                matched_id = None
                for tid,st in tracker_states.items():
                    cx,cy = st['centroid']
                    if np.hypot(cx-centroid[0], cy-centroid[1]) < TRACK_DIST_THR and tid not in used_track_ids:
                        matched_id = tid
                        break
                if matched_id is None:
                    matched_id = next_track_id
                    next_track_id += 1
                    tracker_states[matched_id] = {'centroid': centroid, 'blink': BlinkState(ear_thresh=0.21, consecutive=2)}
                used_track_ids.add(matched_id)
                tracker_states[matched_id]['centroid'] = centroid

                face_crop = frame[y1:y2, x1:x2].copy()
                if face_crop.size == 0:
                    continue

                alive_lbp = simple_texture_check(face_crop, std_thresh=args.lbp_thresh)
                alive_blink = False

                if args.use_blink and mp_landmarks:
                    matched_mesh = None
                    for mesh_coords in mp_landmarks:
                        xs = (mesh_coords[:,0]*frame.shape[1]).astype(int)
                        ys = (mesh_coords[:,1]*frame.shape[0]).astype(int)
                        if (x1<=xs[33]<=x2 and y1<=ys[33]<=y2) or (x1<=xs[362]<=x2 and y1<=ys[362]<=y2):
                            matched_mesh = mesh_coords
                            break
                    if matched_mesh is not None:
                        left_eye_pts = np.array([[matched_mesh[idx][0]*frame.shape[1], matched_mesh[idx][1]*frame.shape[0]] for idx in MP_LEFT_EYE], dtype=np.float32)
                        right_eye_pts = np.array([[matched_mesh[idx][0]*frame.shape[1], matched_mesh[idx][1]*frame.shape[0]] for idx in MP_RIGHT_EYE], dtype=np.float32)
                        ear = (eye_aspect_ratio(left_eye_pts)+eye_aspect_ratio(right_eye_pts))/2.0
                        bs = tracker_states[matched_id]['blink']
                        blinked = bs.update(ear)
                        alive_blink = blinked or (ear>0.18)
                if not args.use_blink:
                    alive_blink = True

                alive = alive_lbp and alive_blink

                if alive:
                    emb = recognizer(frame, kps).astype(np.float32)
                    emb = emb / (np.linalg.norm(emb)+1e-9)
                    faiss.normalize_L2(emb.reshape(1,-1))
                    D,I = faiss_index.search(emb.reshape(1,-1),1)
                    sim = float(D[0][0])
                    name = employee_names[I[0][0]] if sim>args.similarity_thresh else "Unknown"

                    color = colors.get(name,(0,255,0)) if name!="Unknown" else (255,0,0)
                    draw_fancy_bbox(frame, bbox, similarity=sim, name=name, color=color)

                    if name!="Unknown":
                        logger.log(name)
                        enqueue_attendance(employee_id=name, similarity=sim, source="camera_1")
                        try:
                            log_attendance_to_django(name, similarity=sim)
                        except Exception as e:
                            logging.warning(f"Django sync failed for {name}: {e}")
                else:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame,"Spoof?",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

            tracker_states = {tid: tracker_states[tid] for tid in used_track_ids}
            out.write(frame)
            cv2.imshow("Liveness+FAISS Attendance", frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
    finally:
        cap.release()
        out.release()
        logger.close()
        cv2.destroyAllWindows()
        logging.info("Stopped and cleaned up.")

if __name__=="__main__":
    main()
