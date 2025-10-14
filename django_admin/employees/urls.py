# employees/urls.py

from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.employee_list, name='employee_list'),
    path('add/', views.employee_add, name='employee_add'),
    path('delete/<int:emp_id>/', views.employee_delete, name='employee_delete'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
