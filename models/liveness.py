# import cv2
# import numpy as np
# from collections import deque


# # ----------------- Eye-blink detection ----------------- #
# class BlinkDetector:
#     """
#     Kiểm tra mắt nháy dựa trên EAR (Eye Aspect Ratio)
#     - ear_thresh: ngưỡng mở mắt (thường 0.21-0.25)
#     - consecutive_frames: số frame liên tiếp mắt nhắm để tính blink
#     """
#     def __init__(self, ear_thresh=0.21, consecutive_frames=2):
#         self.ear_thresh = ear_thresh
#         self.consec_frames = consecutive_frames
#         self.left_queue = deque(maxlen=consecutive_frames)
#         self.right_queue = deque(maxlen=consecutive_frames)
#         self.blinked = False

#     @staticmethod
#     def eye_aspect_ratio(eye):
#         """
#         eye: 6 điểm landmark mắt [x1,y1,...x6,y6]
#         EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
#         """
#         A = np.linalg.norm(eye[1] - eye[5])
#         B = np.linalg.norm(eye[2] - eye[4])
#         C = np.linalg.norm(eye[0] - eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear

#     def update(self, left_eye, right_eye):
#         left_ear = self.eye_aspect_ratio(left_eye)
#         right_ear = self.eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0

#         self.left_queue.append(avg_ear)
#         self.right_queue.append(avg_ear)

#         # Kiểm tra blink
#         if len(self.left_queue) == self.consec_frames:
#             if all(ear < self.ear_thresh for ear in self.left_queue):
#                 self.blinked = True
#                 self.left_queue.clear()
#                 self.right_queue.clear()
#             else:
#                 self.blinked = False
#         return self.blinked

# # ----------------- LBP texture analysis ----------------- #
# def check_lbp(face_crop, ref_hist=None, threshold=0.7):
#     """
#     face_crop: khuôn mặt crop BGR
#     ref_hist: histogram LBP tham chiếu (tuỳ chọn, None để test)
#     threshold: ngưỡng similarity để coi là alive
#     """
#     gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
#     # LBP extraction
#     lbp = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
#     lbp = cv2.resize(lbp, (64,64))
#     lbp = cv2.Laplacian(lbp, cv2.CV_32F)  # simple texture
#     hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0,256])
#     hist = hist / (hist.sum() + 1e-6)

#     if ref_hist is not None:
#         similarity = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
#         return similarity >= threshold
#     else:
#         # Nếu không có ref, chỉ test là texture "khác rời rạc"
#         # Giả lập: hist.std() > ngưỡng nhỏ
#         return hist.std() > 0.01


# liveness_enhanced.py
# Nâng cấp liveness: Blink + LBP + OpticalFlow + optional ONNX classifier
# Usage:
#   from liveness_enhanced import LivenessEnsemble
#   le = LivenessEnsemble(onnx_path=None)  # or provide path to onnx model
#   is_live = le.update(track_id, face_crop_bgr, left_eye_pts, right_eye_pts, timestamp)

import cv2
import numpy as np
from collections import deque, defaultdict
import time

# ---------------- Config ----------------
DEFAULT_EAR_THRESH = 0.21
DEFAULT_EAR_CONSEC = 2
LBP_STD_THRESH = 0.01        # texture std threshold (tune)
LBP_CORR_THRESH = 0.6        # if using reference histogram
MOTION_MAG_THRESH = 0.5      # optical flow mean magnitude threshold (tune)
TEMPORAL_WINDOW = 7          # number of frames to aggregate per track
MIN_VOTES_TO_DECIDE = 4      # require at least this many positive signals among window
ONNX_ENABLED = False

# ---------------- Blink detector (per-track) ----------------
class BlinkStateSmall:
    def __init__(self, ear_thresh=DEFAULT_EAR_THRESH, consecutive=DEFAULT_EAR_CONSEC):
        self.ear_thresh = ear_thresh
        self.consec = consecutive
        self.ears = deque(maxlen=consecutive)
        self.last_blink_at = 0.0
    def update(self, ear):
        self.ears.append(ear)
        if len(self.ears) < self.consec:
            return False
        if all(e < self.ear_thresh for e in self.ears):
            now = time.time()
            if now - self.last_blink_at > 0.5:
                self.last_blink_at = now
                self.ears.clear()
                return True
        return False

# ---------------- LBP texture helper ----------------
def lbp_texture_score(face_crop, std_thresh=LBP_STD_THRESH):
    """Return boolean: True means texture likely real (not flat printed image)."""
    if face_crop is None or face_crop.size == 0:
        return None, {"error": "Empty face crop"}
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64,64))
    # Use Laplacian as fast proxy for texture; you can replace with real LBP operator.
    lap = cv2.Laplacian(small, cv2.CV_32F)
    hist = cv2.calcHist([cv2.convertScaleAbs(lap)], [0], None, [64], [0,256])
    hist = hist / (hist.sum() + 1e-9)
    return float(np.std(hist)) > std_thresh

# ---------------- Optical flow motion check ----------------
def motion_magnitude(prev_gray, cur_gray):
    """Return mean magnitude of dense optical flow between two gray crops."""
    if prev_gray is None or cur_gray is None:
        return 0.0
    # ensure same size
    if prev_gray.shape != cur_gray.shape:
        cur_gray = cv2.resize(cur_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        pyr_scale=0.5, levels=2, winsize=15,
                                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return float(np.mean(mag))

# ---------------- Optional ONNX classifier ----------------
class ONNXLiveness:
    def __init__(self, onnx_path=None):
        self.enabled = False
        if onnx_path is None:
            return
        try:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.enabled = True
        except Exception as e:
            print(f"[ONNX] Failed to load ONNX model: {e}")
            self.enabled = False
    def predict(self, face_crop):
        if not self.enabled:
            return 0.5
        # preprocess: resize to 128x128, normalize 0..1, CHW
        img = cv2.resize(face_crop, (128,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # HWC -> NCHW
        inp = np.transpose(img, (2,0,1))[None, :, :, :].astype(np.float32)
        try:
            out = self.sess.run(None, {self.input_name: inp})
            # assume output[0] is logits or prob shape (1,2) or (1,1)
            o = out[0]
            if o.ndim == 2 and o.shape[1] == 2:
                # softmax prob for class 1 (live)
                p = np.exp(o[0,1]) / (np.exp(o[0,0]) + np.exp(o[0,1]) + 1e-9)
                return float(p)
            elif o.ndim == 2 and o.shape[1] == 1:
                return float(1.0 / (1.0 + np.exp(-o[0,0])))
            elif o.ndim == 1:
                return float(o[0])
            else:
                return 0.5
        except Exception as e:
            print(f"[ONNX] inference failed: {e}")
            return 0.5

# ---------------- Ensemble per-track ----------------
class LivenessEnsemble:
    def __init__(self, onnx_path=None,
                 temporal_window=TEMPORAL_WINDOW,
                 min_votes=MIN_VOTES_TO_DECIDE,
                 motion_thresh=MOTION_MAG_THRESH,
                 lbp_thresh=LBP_STD_THRESH):
        self.blinkers = defaultdict(lambda: BlinkStateSmall())
        self.prev_gray = {}             # track_id -> previous gray crop
        self.buffers = defaultdict(lambda: deque(maxlen=temporal_window))  # store dicts of signal per frame
        self.onnx = ONNXLiveness(onnx_path)
        self.min_votes = min_votes
        self.motion_thresh = motion_thresh
        self.lbp_thresh = lbp_thresh

    def update(self, track_id, face_crop_bgr, left_eye_pts=None, right_eye_pts=None, timestamp=None):
        """
        Called every frame for each tracked face.
        - track_id: unique id for tracker
        - face_crop_bgr: BGR numpy array
        - left_eye_pts/right_eye_pts: Nx2 arrays of eye landmarks in pixel coords (optional)
        Returns: (decision, details)
          - decision: True if live, False if fake/unknown
          - details: dict of individual signals and score
        """
        ts = timestamp or time.time()
        # 1) Blink signal
        blink_sig = False
        if left_eye_pts is not None and right_eye_pts is not None:
            try:
                left_eye = np.array(left_eye_pts, dtype=np.float32)
                right_eye = np.array(right_eye_pts, dtype=np.float32)
                avg_ear = (BlinkStateSmall.eye_aspect_ratio(left_eye) + BlinkStateSmall.eye_aspect_ratio(right_eye)) / 2.0
                blink_sig = self.blinkers[track_id].update(avg_ear)
            except Exception:
                blink_sig = False

        # 2) LBP/texture signal
        lbp_sig = lbp_texture_score(face_crop_bgr, std_thresh=self.lbp_thresh)

        # 3) Motion / optical flow
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        motion_mag = 0.0
        if track_id in self.prev_gray:
            try:
                motion_mag = motion_magnitude(self.prev_gray[track_id], gray)
            except Exception:
                motion_mag = 0.0
        self.prev_gray[track_id] = gray

        motion_sig = motion_mag > self.motion_thresh

        # 4) ONNX classifier (optional, slow) - only run when other signals ambiguous
        onnx_prob = 0.5
        onnx_sig = False
        if self.onnx.enabled:
            try:
                onnx_prob = self.onnx.predict(face_crop_bgr)
                onnx_sig = onnx_prob > 0.5
            except Exception:
                onnx_prob = 0.5
                onnx_sig = False

        # Compose frame-level signals
        frame_signals = {
            "blink": bool(blink_sig),
            "lbp": bool(lbp_sig),
            "motion": bool(motion_sig),
            "motion_mag": float(motion_mag),
            "onnx_prob": float(onnx_prob),
            "ts": float(ts)
        }

        # Append to temporal buffer
        self.buffers[track_id].append(frame_signals)

        # Aggregation: count positive signals across window
        buf = list(self.buffers[track_id])
        blink_votes = sum(1 for f in buf if f["blink"])
        lbp_votes = sum(1 for f in buf if f["lbp"])
        motion_votes = sum(1 for f in buf if f["motion"])
        onnx_votes = sum(1 for f in buf if f["onnx_prob"] > 0.6) if self.onnx.enabled else 0

        # Voting heuristic:
        # Consider live if (blink_votes sufficient) OR (lbp & motion combined) OR (onnx strong)
        score = 0
        score += 2 * (blink_votes >= 1)       # blink is strong indicator
        score += 1 * (lbp_votes >= (len(buf)//2))
        score += 1 * (motion_votes >= (len(buf)//2))
        score += 2 * (onnx_votes >= 1)        # model confident adds weight

        # Decision threshold: >=2 votes
        decision = score >= 2 and len(buf) >= 1

        details = {
            "blink_votes": blink_votes,
            "lbp_votes": lbp_votes,
            "motion_votes": motion_votes,
            "onnx_votes": onnx_votes,
            "score": score,
            "window_len": len(buf),
            "latest_motion_mag": motion_mag,
            "latest_lbp": lbp_sig,
            "latest_blink": blink_sig,
            "latest_onnx_prob": onnx_prob,
        }
        return decision, details

    def clear_track(self, track_id):
        if track_id in self.buffers:
            del self.buffers[track_id]
        if track_id in self.prev_gray:
            del self.prev_gray[track_id]
        if track_id in self.blinkers:
            del self.blinkers[track_id]
