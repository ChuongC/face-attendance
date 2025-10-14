from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import FaceDataset
from admin_panel.models import Employee
from admin_panel.embedding_extractor import get_face_embedding
import numpy as np
import os

EMB_DIR = "embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

@receiver(post_save, sender=FaceDataset)
def generate_face_embedding(sender, instance, created, **kwargs):
    if not created or not instance.image:
        return

    emp = instance.employee
    emb = get_face_embedding(instance.image.path)
    if emb is None:
        print(f"[❌] Không thể tạo embedding cho {emp.name}")
        return

    emb_path = os.path.join(EMB_DIR, f"{emp.employee_id}.npy")
    np.save(emb_path, emb)
    emp.face_embedding_path = emb_path
    emp.save()
    print(f"[✅] Đã tạo embedding cho {emp.name} → {emb_path}")
