from django.shortcuts import render

# Create your views here.
# employees/views.py
import os
import numpy as np
import faiss
import pickle
import cv2
from django.shortcuts import render, redirect
from .forms import EmployeeForm
from .models import Employee
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import SCRFD, ArcFaceONNX  # module sẵn có trong dự án gốc
from utils.helpers import build_embedding  # nếu bạn tách ra từ CLI cũ

EMPLOYEE_DB_PATH = "database/employee_db.pkl"
FAISS_INDEX_PATH = "database/face_index.faiss"

# Load detector & recognizer chỉ 1 lần
detector = SCRFD("./weights/det_10g.onnx")
recognizer = ArcFaceONNX("./weights/w600k_r50.onnx")

def load_db():
    if os.path.exists(EMPLOYEE_DB_PATH):
        with open(EMPLOYEE_DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(EMPLOYEE_DB_PATH, "wb") as f:
        pickle.dump(db, f)

def load_faiss():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_INDEX_PATH + ".names", "rb") as f:
            names = pickle.load(f)
        return index, names
    return None, []

def save_faiss(index, names):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH + ".names", "wb") as f:
        pickle.dump(names, f)

def employee_list(request):
    employees = Employee.objects.all().order_by('-created_at')
    return render(request, 'employees/employee_list.html', {'employees': employees})

def employee_add(request):
    if request.method == 'POST':
        form = EmployeeForm(request.POST, request.FILES)
        if form.is_valid():
            emp = form.save()
            db = load_db()
            index, names = load_faiss()

            imgs = [emp.image1, emp.image2, emp.image3]
            embeddings, new_names = [], []

            for img_field in imgs:
                if not img_field:
                    continue
                img_path = img_field.path
                emb = build_embedding(detector, recognizer, img_path)
                if emb is not None:
                    embeddings.append(emb)
                    new_names.append(emp.code)

            if embeddings:
                emb_np = np.vstack(embeddings).astype(np.float32)
                faiss.normalize_L2(emb_np)
                if index is None:
                    dim = emb_np.shape[1]
                    index = faiss.IndexFlatIP(dim)
                index.add(emb_np)
                names.extend(new_names)

            db[emp.code] = [img.path for img in imgs if img]
            save_db(db)
            save_faiss(index, names)

            return redirect('employee_list')
    else:
        form = EmployeeForm()
    return render(request, 'employees/employee_form.html', {'form': form})

def employee_delete(request, emp_id):
    emp = Employee.objects.get(id=emp_id)
    emp.delete()
    # TODO: cập nhật lại FAISS và DB pickle sau khi xóa
    return redirect('employee_list')
