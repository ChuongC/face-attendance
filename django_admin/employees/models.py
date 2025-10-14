# employees/models.py
from django.db import models

class FaceDataset(models.Model):
    """
    Bảng lưu các ảnh khuôn mặt gốc của nhân viên (nhiều ảnh cho 1 người).
    Khi có ảnh mới -> sinh embedding tự động.
    """
    # Liên kết tới bảng trung tâm trong admin_panel
    employee = models.ForeignKey(
        'admin_panel.Employee', 
        on_delete=models.CASCADE,
        related_name='face_images'
    )
    image = models.ImageField(upload_to='faces/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.employee.name} ({self.employee.employee_id})"
