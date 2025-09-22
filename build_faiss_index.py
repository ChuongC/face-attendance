import os
import cv2
import faiss
import pickle
import numpy as np
from models import ArcFaceONNX, SCRFD

# Cấu hình
FACES_DIR = "./faces"
DET_WEIGHT = "./weights/det_10g.onnx"
REC_WEIGHT = "./weights/w600k_r50.onnx"
FAISS_INDEX_PATH = "./database/face_database/faiss_index.bin"
EMPLOYEE_DB_PATH = "./database/face_database/employee_db.pkl"

# Khởi tạo detector + recognizer
detector = SCRFD(DET_WEIGHT)
recognizer = ArcFaceONNX(REC_WEIGHT)

# Chuẩn bị dữ liệu
embeddings = []
names = []

for filename in os.listdir(FACES_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    name = os.path.splitext(filename)[0]  # Lấy tên file làm tên nhân viên
    img_path = os.path.join(FACES_DIR, filename)
    image = cv2.imread(img_path)
    
    bboxes, kpss = detector.detect(image, input_size=(640, 640), thresh=0.5, max_num=1)
    if len(kpss) == 0:
        print(f"No face detected in {img_path}, skipping.")
        continue
    
    emb = recognizer(image, kpss[0])
    embeddings.append(emb.astype(np.float32))
    names.append(name)
    print(f"Added {name} to FAISS index.")

if len(embeddings) == 0:
    raise ValueError("No faces found to build index.")

embeddings_np = np.vstack(embeddings)

# Tạo FAISS index (IVF + PQ có thể dùng cho nhiều người, ở đây demo IndexFlatL2)
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)  # Dùng inner product để thay cosine similarity
faiss.normalize_L2(embeddings_np)     # Chuẩn hóa vector
index.add(embeddings_np)

# Lưu index
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"FAISS index saved to {FAISS_INDEX_PATH}")

# Lưu mapping tên nhân viên
with open(EMPLOYEE_DB_PATH, "wb") as f:
    pickle.dump(names, f)
print(f"Employee DB saved to {EMPLOYEE_DB_PATH}")
