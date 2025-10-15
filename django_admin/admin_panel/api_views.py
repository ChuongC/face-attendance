# admin_panel/views.py
from datetime import timedelta
from django.utils.timezone import now
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import AttendanceRecord, Employee

@api_view(['POST'])
def log_attendance(request):
    emp_id = request.data.get("employee_id")
    timestamp = request.data.get("timestamp")
    similarity = request.data.get("similarity", 1.0)
    source = request.data.get("source", "camera")

    try:
        emp = Employee.objects.get(employee_id=emp_id)
    except Employee.DoesNotExist:
        return Response({"error": f"Employee '{emp_id}' not found"}, status=404)

    # 🔹 Lấy bản ghi gần nhất
    last_record = AttendanceRecord.objects.filter(employee_name=emp.name).order_by('-check_in_time').first()
    now_time = now()

    # 🔹 Xác định trạng thái
    if last_record and (now_time - last_record.check_in_time) < timedelta(minutes=5):
        return Response({"message": "Duplicate detected, ignored."}, status=200)

    status = "Check-in"
    if last_record and (now_time - last_record.check_in_time) < timedelta(hours=4):
        status = "Check-out"

    # 🔹 Lưu bản ghi mới
    AttendanceRecord.objects.create(
        employee_name=emp.name,
        check_in_time=now_time,
        status=status
    )

    print(f"[SYNC ✅] Logged {status} for {emp.name}")
    return Response({"message": f"Logged {status} for {emp.name}"}, status=200)
