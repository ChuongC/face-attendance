from django.db.models.signals import post_save
from django.dispatch import receiver
from admin_panel.models import AttendanceRecord
from .webhook import send_to_hrm

@receiver(post_save, sender=AttendanceRecord)
def auto_push_to_hrm(sender, instance, created, **kwargs):
    """
    Khi có AttendanceRecord mới → tự động gửi webhook sang HRM.
    """
    if created:
        event_type = "check-in" if instance.status == "Check-in" else "check-out"
        try:
            send_to_hrm(
                employee_id=instance.employee.employee_id,
                event_type=event_type,
                check_time=instance.check_in_time,
            )
        except Exception as e:
            print(f"[HRM ⚠] Error pushing to HRM: {e}")
