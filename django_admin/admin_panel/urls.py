from django.urls import path
from . import views
from .api_views import log_attendance

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path('upload/<str:employee_id>/', views.upload_photo, name='upload_photo'),
    path("employees/", views.employees_list, name="employees_list"),
    path("attendance/", views.attendance_list, name="attendance_list"),
    path('api/log_attendance/', log_attendance, name='log_attendance')
]
