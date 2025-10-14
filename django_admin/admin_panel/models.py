from django.db import models

# Create your models here.

from django.db import models
from datetime import datetime

class Employee(models.Model):
    """
    Bảng nhân viên - đồng bộ với employee_db.pkl
    """
    employee_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100)
    department = models.CharField(max_length=100, blank=True, null=True)
    position = models.CharField(max_length=100, blank=True, null=True)
    face_embedding_path = models.CharField(max_length=255, blank=True, null=True)
    photo = models.ImageField(upload_to='', blank=True, null=True)
    date_created = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return f"{self.name} ({self.employee_id})"


class AttendanceRecord(models.Model):
    """
    Bảng chấm công - đồng bộ với attendance_logger.py
    """
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    check_in_time = models.DateTimeField(default=datetime.now)
    similarity = models.FloatField(default=0.0)
    source = models.CharField(max_length=100, default='camera_1')

    def __str__(self):
        return f"{self.employee.name} - {self.check_in_time.strftime('%Y-%m-%d %H:%M:%S')}"
