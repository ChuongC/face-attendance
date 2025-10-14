import cv2
import numpy as np

from django.conf import settings

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import SCRFD, ArcFaceONNX  # module sẵn có trong dự án gốc

# Load detector & recognizer chỉ 1 lần
detector = SCRFD("./weights/det_10g.onnx")
recognizer = ArcFaceONNX("./weights/w600k_r50.onnx")

def get_face_embedding(image_path):
    """Trích xuất embedding khuôn mặt từ ảnh."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Không đọc được ảnh: {image_path}")
        return None

    try:
        bboxes, kpss = detector.detect(img, input_size=(640, 640))
        if bboxes is None or len(kpss) == 0:
            print(f"[WARNING] ❌ Không phát hiện khuôn mặt trong {image_path}")
            return None

        # Dùng đúng cách như trong CLI
        emb = recognizer(img, kpss[0]).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
    except Exception as e:
        print(f"[ERROR] Lỗi khi trích xuất embedding: {e}")
        return None
