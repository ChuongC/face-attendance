# employee_utils.py
import os
import cv2
import faiss
import pickle
import numpy as np
import logging
from typing import List

from models import SCRFD, ArcFaceONNX

EMPLOYEE_DB_PATH = "database/employee_db.pkl"
FAISS_INDEX_PATH = "database/face_index.faiss"

# ---------------- Load/Save ----------------
def load_employee_db():
    if os.path.exists(EMPLOYEE_DB_PATH):
        with open(EMPLOYEE_DB_PATH, "rb") as f:
            db = pickle.load(f)
    else:
        db = {}
    return db

def save_employee_db(db):
    with open(EMPLOYEE_DB_PATH, "wb") as f:
        pickle.dump(db, f)

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_INDEX_PATH + ".names", "rb") as f:
            names = pickle.load(f)
    else:
        index = None
        names = []
    return index, names

def save_faiss_index(index, names):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + ".names", "wb") as f:
        pickle.dump(names, f)

# ---------------- Embedding ----------------
def build_embedding(detector, recognizer, img_bytes: bytes):
    import numpy as np
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img is None:
        return None
    bboxes, kpss = detector.detect(img, input_size=(320,320), thresh=0.5, max_num=1)
    if len(kpss) == 0:
        return None
    emb = recognizer(img, kpss[0]).astype(np.float32)
    emb = emb / (np.linalg.norm(emb)+1e-9)
    return emb

# ---------------- API Functions ----------------
def add_employee(detector, recognizer, name: str, images: List[bytes]):
    db = load_employee_db()
    index, names = load_faiss_index()

    if name in db:
        db[name].extend([f"{name}_{len(db[name])+i}.jpg" for i in range(len(images))])
    else:
        db[name] = [f"{name}_{i}.jpg" for i in range(len(images))]

    embeddings = []
    new_names = []
    for img_bytes in images:
        emb = build_embedding(detector, recognizer, img_bytes)
        if emb is not None:
            embeddings.append(emb)
            new_names.append(name)
    if len(embeddings) == 0:
        return {"status": "error", "message": "No valid embeddings created."}

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
    return {"status": "success", "message": f"Added/Updated {name}", "embeddings": len(embeddings)}

def remove_employee(detector, recognizer, name: str):
    db = load_employee_db()
    index, names = load_faiss_index()

    if name not in db:
        return {"status": "error", "message": f"{name} not found."}

    del db[name]

    all_embeddings = []
    all_names = []
    for n, imgs in db.items():
        for img_path in imgs:
            emb = build_embedding(detector, recognizer, open(img_path, "rb").read())
            if emb is not None:
                all_embeddings.append(emb)
                all_names.append(n)
    if len(all_embeddings) > 0:
        emb_np = np.vstack(all_embeddings)
        faiss.normalize_L2(emb_np)
        dim = emb_np.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_np)
    else:
        index = None
        all_names = []

    save_employee_db(db)
    save_faiss_index(index, all_names)
    return {"status": "success", "message": f"Removed {name}"}

def list_employees():
    db = load_employee_db()
    return [{"name": name, "images": len(imgs)} for name, imgs in db.items()]
