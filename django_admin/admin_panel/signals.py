# import os
# import cv2
# import numpy as np
# from django.db.models.signals import post_save
# from django.dispatch import receiver
# from employees.models import EmployeeImage
# from .models import Employee
# from models.ArcFaceONNX import ArcFaceONNX

# @receiver(post_save, sender=EmployeeImage)
# def create_face_embedding(sender, instance, created, **kwargs):
#     """
#     Khi thêm ảnh mới => sinh embedding => lưu file .npy => update Employee
#     """
#     if not created:
#         return

#     employee = instance.employee
#     img_path = instance.image.path
#     if not os.path.exists(img_path):
#         return

#     # Tạo thư mục lưu embeddings nếu chưa có
#     os.makedirs("embeddings", exist_ok=True)

#     # Tải mô hình ArcFace
#     recognizer = ArcFaceONNX("./weights/w600k_r50.onnx")

#     # Sinh embedding
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARN] Không thể đọc ảnh {img_path}")
#         return

#     emb = recognizer.get(img).astype(np.float32)
#     emb_path = f"embeddings/{employee.employee_id}.npy"
#     np.save(emb_path, emb)

#     # Cập nhật Employee
#     employee.face_embedding_path = emb_path
#     employee.save(update_fields=["face_embedding_path"])
#     print(f"[SYNC] Tạo embedding cho {employee.name} → {emb_path}")
