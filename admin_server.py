import os
import cv2
import faiss
import pickle
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, Form
from typing import List

from models import SCRFD, ArcFaceONNX

EMPLOYEE_DB_PATH = "database/employee_db.pkl"
FAISS_INDEX_PATH = "database/face_index.faiss"

# ----------------- Core Utils -----------------
def load_employee_db():
    return pickle.load(open(EMPLOYEE_DB_PATH, "rb")) if os.path.exists(EMPLOYEE_DB_PATH) else {}

def save_employee_db(db):
    with open(EMPLOYEE_DB_PATH, "wb") as f:
        pickle.dump(db, f)

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        names = pickle.load(open(FAISS_INDEX_PATH + ".names", "rb"))
    else:
        index, names = None, []
    return index, names

def save_faiss_index(index, names):
    faiss.write_index(index, FAISS_INDEX_PATH)
    pickle.dump(names, open(FAISS_INDEX_PATH + ".names", "wb"))

def build_embedding(detector, recognizer, file_bytes):
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return None
    bboxes, kpss = detector.detect(img, input_size=(320,320), thresh=0.5, max_num=1)
    if len(kpss) == 0:
        return None
    emb = recognizer(img, kpss[0]).astype(np.float32)
    return emb / (np.linalg.norm(emb)+1e-9)

# ----------------- FastAPI -----------------
app = FastAPI()
detector = SCRFD("./weights/det_10g.onnx")
recognizer = ArcFaceONNX("./weights/w600k_r50.onnx")

@app.get("/")
def root():
    return {"msg": "Face Attendance Admin API running"}


@app.post("/employee/add")
async def add_employee(name: str = Form(...), files: List[UploadFile] = []):
    db = load_employee_db()
    index, names = load_faiss_index()

    if name not in db:
        db[name] = []

    embeddings, new_names = [], []
    for file in files:
        content = await file.read()
        emb = build_embedding(detector, recognizer, content)
        if emb is not None:
            embeddings.append(emb)
            new_names.append(name)
            # Lưu ảnh vào thư mục để rollback sau này
            save_path = f"database/images/{name}_{file.filename}"
            with open(save_path, "wb") as f:
                f.write(content)
            db[name].append(save_path)

    if embeddings:
        emb_np = np.vstack(embeddings)
        faiss.normalize_L2(emb_np)
        if index is None:
            dim = emb_np.shape[1]
            index = faiss.IndexFlatIP(dim)
            names = []
        index.add(emb_np)
        names.extend(new_names)

    save_employee_db(db)
    save_faiss_index(index, names)
    return {"status": "ok", "added": len(embeddings)}

@app.delete("/employee/remove")
def remove_employee(name: str):
    db = load_employee_db()
    if name not in db:
        return {"status": "error", "msg": "Not found"}
    del db[name]

    # rebuild FAISS
    index, names = None, []
    all_embeddings = []
    for n, imgs in db.items():
        for p in imgs:
            emb = build_embedding(detector, recognizer, open(p, "rb").read())
            if emb is not None:
                all_embeddings.append(emb)
                names.append(n)
    if all_embeddings:
        emb_np = np.vstack(all_embeddings)
        faiss.normalize_L2(emb_np)
        index = faiss.IndexFlatIP(emb_np.shape[1])
        index.add(emb_np)

    save_employee_db(db)
    save_faiss_index(index, names)
    return {"status": "ok", "removed": name}

@app.get("/employee/list")
def list_employees():
    db = load_employee_db()
    return {"employees": {k: len(v) for k, v in db.items()}}

