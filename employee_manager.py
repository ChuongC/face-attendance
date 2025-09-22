"""
Employee Manager CLI với menu
- Thêm/xóa/liệt kê nhân viên
- Tích hợp FAISS + DB
- Chuẩn bị pipeline thực tế cho admin server
"""

import os
import cv2
import faiss
import pickle
import numpy as np
import logging

from models import SCRFD, ArcFaceONNX

EMPLOYEE_DB_PATH = "database/employee_db.pkl"
FAISS_INDEX_PATH = "database/face_index.faiss"

# ---------------- Logging ----------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# ---------------- Build embedding ----------------
def build_embedding(detector, recognizer, img_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Cannot read image: {img_path}")
        return None
    bboxes, kpss = detector.detect(img, input_size=(320,320), thresh=0.5, max_num=1)
    if len(kpss) == 0:
        logging.warning(f"No face detected in image: {img_path}")
        return None
    emb = recognizer(img, kpss[0]).astype(np.float32)
    emb = emb / (np.linalg.norm(emb)+1e-9)
    return emb

# ---------------- Commands ----------------
def add_employee(detector, recognizer):
    db = load_employee_db()
    index, names = load_faiss_index()

    name = input("Enter employee name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    imgs_input = input("Enter paths to images (space-separated): ").strip()
    img_paths = imgs_input.split()
    if len(img_paths) == 0:
        print("No images provided.")
        return

    if name in db:
        print(f"{name} exists, appending images.")
        db[name].extend(img_paths)
    else:
        db[name] = img_paths

    embeddings = []
    new_names = []
    for p in img_paths:
        emb = build_embedding(detector, recognizer, p)
        if emb is not None:
            embeddings.append(emb)
            new_names.append(name)
    if len(embeddings) == 0:
        print("No valid embeddings created.")
        return

    # Update FAISS
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
    print(f"Added/Updated {name} with {len(embeddings)} embeddings.")

def remove_employee(detector, recognizer):
    db = load_employee_db()
    index, names = load_faiss_index()
    name = input("Enter employee name to remove: ").strip()
    if name not in db:
        print(f"{name} not found.")
        return
    del db[name]

    # Rebuild FAISS from remaining DB
    print("Rebuilding FAISS index...")
    all_embeddings = []
    all_names = []
    for n, imgs in db.items():
        for img_path in imgs:
            emb = build_embedding(detector, recognizer, img_path)
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
    print(f"Removed {name} successfully.")

def list_employees():
    db = load_employee_db()
    if len(db) == 0:
        print("No employees in DB.")
        return
    print("Employees:")
    for name, imgs in db.items():
        print(f"- {name}: {len(imgs)} image(s)")

# ---------------- Main menu ----------------
def main():
    setup_logging()
    detector = SCRFD("./weights/det_10g.onnx")
    recognizer = ArcFaceONNX("./weights/w600k_r50.onnx")

    while True:
        print("\n====== Employee Management Menu ======")
        print("1. Add employee")
        print("2. Remove employee")
        print("3. List employees")
        print("4. Exit")
        choice = input("Select an option [1-4]: ").strip()

        if choice == "1":
            add_employee(detector, recognizer)
        elif choice == "2":
            remove_employee(detector, recognizer)
        elif choice == "3":
            list_employees()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
