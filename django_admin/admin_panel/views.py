from django.shortcuts import render, redirect, get_object_or_404
from .models import Employee, AttendanceRecord
from .forms import EmployeePhotoUploadForm
from .sync_utils import sync_from_django_to_faiss
import numpy as np, cv2, logging

logger = logging.getLogger(__name__)

def dashboard(request):
    employees = Employee.objects.all().order_by('employee_id')
    attendance_records = AttendanceRecord.objects.select_related('employee').order_by('-check_in_time')[:10]
    return render(request, 'dashboard.html', {'employees': employees, 'attendance_records': attendance_records})

def employees_list(request):
    employees = Employee.objects.all().order_by("name")
    return render(request, "employees.html", {"employees": employees})

def attendance_list(request):
    records = AttendanceRecord.objects.select_related("employee").order_by("-check_in_time")[:100]
    return render(request, "attendance.html", {"records": records})

def upload_photo(request, employee_id):
    emp = get_object_or_404(Employee, employee_id=employee_id)
    if request.method == "POST":
        form = EmployeePhotoUploadForm(request.POST, request.FILES, instance=emp)
        if form.is_valid():
            emp = form.save()
            img_file = request.FILES.get('photo')
            if img_file:
                try:
                    file_bytes = np.frombuffer(img_file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    from .embedding_extractor import get_face_embedding
                    emb = get_face_embedding(img)
                    if emb is not None:
                        sync_from_django_to_faiss(emp.employee_id, emb)
                        logger.info(f"[SYNC ✅] Employee {emp.employee_id} synced to FAISS")
                except Exception as e:
                    logger.error(f"[SYNC ❌] {e}")
            return redirect('dashboard')
    else:
        form = EmployeePhotoUploadForm(instance=emp)
    return render(request, 'upload_photo.html', {'form': form, 'employee': emp})
