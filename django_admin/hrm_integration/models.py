from django.db import models

class HRMWebhookLog(models.Model):
    employee_id = models.CharField(max_length=100)
    event_type = models.CharField(max_length=20)  # check-in / check-out
    payload = models.JSONField()
    status_code = models.IntegerField(null=True, blank=True)
    response = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.employee_id} - {self.event_type} ({self.timestamp})"
