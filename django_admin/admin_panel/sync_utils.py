import os
import pickle
import faiss
import numpy as np
from django.conf import settings
from .models import Employee

DB_DIR = os.path.join(settings.BASE_DIR, '..', 'database')
EMP_DB_PATH = os.path.join(DB_DIR, 'employee_db.pkl')
FACE_INDEX_PATH = os.path.join(DB_DIR, 'face_index.faiss')


def load_employee_db():
    if os.path.exists(EMP_DB_PATH):
        with open(EMP_DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_employee_db(db):
    with open(EMP_DB_PATH, 'wb') as f:
        pickle.dump(db, f)


def load_faiss_index(dimension=512):
    if os.path.exists(FACE_INDEX_PATH):
        return faiss.read_index(FACE_INDEX_PATH)
    index = faiss.IndexFlatL2(dimension)
    return index


def save_faiss_index(index):
    faiss.write_index(index, FACE_INDEX_PATH)


def sync_from_django_to_faiss(emp_id=None, embedding=None):
    """
    Cập nhật FAISS khi:
    - Có embedding mới được thêm từ GUI (emp_id, embedding)
    - Hoặc chạy đồng bộ toàn bộ (nếu không truyền tham số)
    """
    emp_db = load_employee_db()
    index = load_faiss_index()

    # === 1️⃣ Trường hợp: upload ảnh khuôn mặt trực tiếp trong GUI ===
    if emp_id and embedding is not None:
        emp = Employee.objects.filter(employee_id=emp_id).first()
        if emp:
            # Nếu đã tồn tại -> cập nhật embedding mới
            if emp_id in emp_db:
                idx = list(emp_db.keys()).index(emp_id)
                index.reconstruct(idx)  # reconstruct gọi để tránh duplicate
                # Xóa embedding cũ ra khỏi index nếu cần
                index.remove_ids(np.array([idx], dtype=np.int64))
            
            # Thêm embedding mới vào FAISS
            index.add(embedding.reshape(1, -1))
            
            # Cập nhật employee_db
            emp_db[emp_id] = {
                'name': emp.name,
                'department': emp.department,
                'position': emp.position,
                'photo': emp.photo.url if emp.photo else None
            }
            print(f"[INFO] Synced employee {emp_id} → FAISS index.")
        
        save_faiss_index(index)
        save_employee_db(emp_db)
        return

    # === 2️⃣ Trường hợp: đồng bộ toàn bộ dữ liệu khi khởi tạo ===
    for emp in Employee.objects.all():
        if emp.employee_id not in emp_db:
            if emp.photo and os.path.exists(emp.photo.path):
                from embedding_extractor import get_face_embedding
                import cv2
                img = cv2.imread(emp.photo.path)
                emb = get_face_embedding(img)
                if emb is not None:
                    index.add(emb.reshape(1, -1))
                    emp_db[emp.employee_id] = {
                        'name': emp.name,
                        'department': emp.department,
                        'position': emp.position,
                        'photo': emp.photo.url
                    }
                    print(f"[INFO] Synced {emp.name} ({emp.employee_id}) to FAISS index.")

    save_faiss_index(index)
    save_employee_db(emp_db)

import os
import django

# Cấu hình Django settings để ORM có thể hoạt động
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_server.settings')


from admin_panel.models import Employee, AttendanceRecord

def log_attendance_to_django(employee_name, similarity=1.0, source='camera_1'):
    """
    Ghi dữ liệu chấm công trực tiếp vào Django database.
    """
    try:
        emp = Employee.objects.get(name=employee_name)
        AttendanceRecord.objects.create(
            employee=emp,
            similarity=similarity,
            source=source
        )
        print(f"[SYNC ✅] Logged attendance for {employee_name} to Django DB.")
    except Employee.DoesNotExist:
        print(f"[SYNC ⚠] Employee '{employee_name}' not found in Django DB.")
    except Exception as e:
        print(f"[SYNC ❌] Error syncing to Django: {e}")
