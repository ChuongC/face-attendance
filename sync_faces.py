import os
import shutil

# Đường dẫn tới hai thư mục
DJANGO_FACES_DIR = "django_admin/faces"
MAIN_FACES_DIR = "faces"

def sync_faces():
    """
    Đồng bộ ảnh nhân viên từ django_admin/faces sang faces/
    """
    if not os.path.exists(DJANGO_FACES_DIR):
        print(f"❌ Không tìm thấy thư mục: {DJANGO_FACES_DIR}")
        return

    os.makedirs(MAIN_FACES_DIR, exist_ok=True)

    for employee_name in os.listdir(DJANGO_FACES_DIR):
        src_dir = os.path.join(DJANGO_FACES_DIR, employee_name)
        dst_dir = os.path.join(MAIN_FACES_DIR, employee_name)

        # Nếu là thư mục nhân viên
        if os.path.isdir(src_dir):
            if os.path.exists(dst_dir):
                # Ghi đè file cũ
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"✅ Đã đồng bộ: {employee_name}")

if __name__ == "__main__":
    sync_faces()
