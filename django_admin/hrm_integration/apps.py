from django.apps import AppConfig

class HrmIntegrationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "hrm_integration"

    def ready(self):
        import hrm_integration.signals  # nạp signals
