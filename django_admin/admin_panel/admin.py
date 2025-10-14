from django.contrib import admin
from .models import Employee, AttendanceRecord
from .sync_utils import sync_from_django_to_faiss
import numpy as np  
from django.conf import settings
import os

from .models import Employee
@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'name', 'department', 'position', 'date_created')
    search_fields = ('name', 'employee_id', 'department')

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        obj.refresh_from_db()

        if obj.photo and hasattr(obj.photo, 'path'):
            from .embedding_extractor import get_face_embedding
            from .sync_utils import sync_from_django_to_faiss

            img_path = obj.photo.path
            print(f"[DEBUG] Đọc ảnh từ: {img_path}")

            if os.path.exists(img_path):
                embedding = get_face_embedding(img_path)  # ✅ truyền đường dẫn, không dùng imread ở đây

                if embedding is not None:
                    # 1️⃣ Lưu embedding ra file .npy
                    embedding_dir = os.path.join(settings.MEDIA_ROOT, 'embeddings')
                    os.makedirs(embedding_dir, exist_ok=True)
                    embedding_path = os.path.join(embedding_dir, f"{obj.employee_id}.npy")
                    np.save(embedding_path, embedding)
                    obj.face_embedding_path = os.path.relpath(embedding_path, settings.MEDIA_ROOT)
                    obj.save(update_fields=['face_embedding_path'])

                    # 2️⃣ Đồng bộ FAISS
                    sync_from_django_to_faiss(obj.employee_id, embedding)

                    self.message_user(request, "✅ Đã trích xuất, lưu embedding và đồng bộ FAISS thành công.")
                else:
                    self.message_user(request, "❌ Không phát hiện khuôn mặt trong ảnh.", level='error')
            else:
                self.message_user(request, f"Không tìm thấy ảnh: {img_path}", level='error')
@admin.register(AttendanceRecord)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('employee', 'check_in_time', 'similarity', 'source')
    search_fields = ('employee__name',)
