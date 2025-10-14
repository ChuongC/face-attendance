from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Employee, AttendanceRecord

@csrf_exempt
def log_attendance(request):
    """
    Endpoint để nhận dữ liệu chấm công từ hệ thống khác (HRM/Payroll)
    POST JSON: { "employee_id": "E001", "timestamp": "2025-10-12T09:30:00", "similarity": 0.87 }
    """
    import json
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            emp_id = data.get("employee_id")
            emp = Employee.objects.get(employee_id=emp_id)
            AttendanceRecord.objects.create(
                employee=emp,
                similarity=data.get("similarity", 1.0),
                source="external_api"
            )
            return JsonResponse({"status": "ok"})
        except Employee.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Employee not found"}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "Invalid method"}, status=405)
