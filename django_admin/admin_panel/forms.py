from django import forms
from .models import Employee

class EmployeePhotoUploadForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['photo']  
