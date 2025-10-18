import requests
from django.conf import settings
from .models import HRMWebhookLog

# Địa chỉ Webhook HRM (có thể cấu hình trong settings.py)
HRM_WEBHOOK_URL = getattr(settings, "HRM_WEBHOOK_URL", "http://localhost:9000/hrm/webhook")

def send_to_hrm(employee_id, event_type, check_time):
    """
    Gửi dữ liệu chấm công sang hệ thống HRM qua webhook HTTP POST.
    """
    payload = {
        "employee_id": employee_id,
        "event_type": event_type,
        "timestamp": check_time.isoformat(),
    }
    try:
        resp = requests.post(HRM_WEBHOOK_URL, json=payload, timeout=5)
        HRMWebhookLog.objects.create(
            employee_id=employee_id,
            event_type=event_type,
            payload=payload,
            status_code=resp.status_code,
            response=resp.text,
        )
        print(f"[HRM ✅] Sent {employee_id} ({event_type}) → {HRM_WEBHOOK_URL}")
        return resp.status_code
    except Exception as e:
        HRMWebhookLog.objects.create(
            employee_id=employee_id,
            event_type=event_type,
            payload=payload,
            response=str(e),
        )
        print(f"[HRM ❌] Failed to send {employee_id}: {e}")
        return None
