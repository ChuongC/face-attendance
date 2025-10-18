from django.urls import path
from . import views
from .api_views import log_attendance
from django.http import JsonResponse
from hrm_integration.webhook import send_to_hrm
from datetime import datetime

def test_hrm(request):
    send_to_hrm("test123", "check-in", datetime.now())
    return JsonResponse({"status": "ok"})

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path('upload/<str:employee_id>/', views.upload_photo, name='upload_photo'),
    path("employees/", views.employees_list, name="employees_list"),
    path("attendance/", views.attendance_list, name="attendance_list"),
    path('api/log_attendance/', log_attendance, name='log_attendance'),
    path("api/test_hrm/", test_hrm)
]
