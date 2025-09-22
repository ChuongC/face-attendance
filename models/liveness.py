import cv2
import numpy as np
from collections import deque


# ----------------- Eye-blink detection ----------------- #
class BlinkDetector:
    """
    Kiểm tra mắt nháy dựa trên EAR (Eye Aspect Ratio)
    - ear_thresh: ngưỡng mở mắt (thường 0.21-0.25)
    - consecutive_frames: số frame liên tiếp mắt nhắm để tính blink
    """
    def __init__(self, ear_thresh=0.21, consecutive_frames=2):
        self.ear_thresh = ear_thresh
        self.consec_frames = consecutive_frames
        self.left_queue = deque(maxlen=consecutive_frames)
        self.right_queue = deque(maxlen=consecutive_frames)
        self.blinked = False

    @staticmethod
    def eye_aspect_ratio(eye):
        """
        eye: 6 điểm landmark mắt [x1,y1,...x6,y6]
        EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
        """
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def update(self, left_eye, right_eye):
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        self.left_queue.append(avg_ear)
        self.right_queue.append(avg_ear)

        # Kiểm tra blink
        if len(self.left_queue) == self.consec_frames:
            if all(ear < self.ear_thresh for ear in self.left_queue):
                self.blinked = True
                self.left_queue.clear()
                self.right_queue.clear()
            else:
                self.blinked = False
        return self.blinked

# ----------------- LBP texture analysis ----------------- #
def check_lbp(face_crop, ref_hist=None, threshold=0.7):
    """
    face_crop: khuôn mặt crop BGR
    ref_hist: histogram LBP tham chiếu (tuỳ chọn, None để test)
    threshold: ngưỡng similarity để coi là alive
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # LBP extraction
    lbp = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lbp = cv2.resize(lbp, (64,64))
    lbp = cv2.Laplacian(lbp, cv2.CV_32F)  # simple texture
    hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0,256])
    hist = hist / (hist.sum() + 1e-6)

    if ref_hist is not None:
        similarity = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
        return similarity >= threshold
    else:
        # Nếu không có ref, chỉ test là texture "khác rời rạc"
        # Giả lập: hist.std() > ngưỡng nhỏ
        return hist.std() > 0.01
