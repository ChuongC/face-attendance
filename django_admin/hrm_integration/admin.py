from django.contrib import admin
from .models import HRMWebhookLog

@admin.register(HRMWebhookLog)
class HRMWebhookLogAdmin(admin.ModelAdmin):
    list_display = ("employee_id", "event_type", "status_code", "timestamp")
    search_fields = ("employee_id", "event_type", "response")
    list_filter = ("event_type", "status_code")
    ordering = ("-timestamp",)
