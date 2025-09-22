import csv
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.db_utils import fetch_all_attendance

def show_attendance():
    records = fetch_all_attendance()
    if not records:
        print("⚠️ No attendance records found.")
        return
    print("\n=== Attendance Records ===")
    for row in records:
        print(f"ID: {row[0]} | Name: {row[1]} | Timestamp: {row[2]}")

def export_csv(file_path: str = "./attendance_export.csv"):
    records = fetch_all_attendance()
    if not records:
        print("⚠️ No records to export.")
        return
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Timestamp"])
        writer.writerows(records)
    print(f"✅ Attendance exported to {file_path}")

def export_excel(file_path: str = "./attendance_export.xlsx"):
    records = fetch_all_attendance()
    if not records:
        print("⚠️ No records to export.")
        return
    df = pd.DataFrame(records, columns=["ID", "Name", "Timestamp"])
    df.to_excel(file_path, index=False)
    print(f"✅ Attendance exported to {file_path}")

if __name__ == "__main__":
    print("1. Show attendance")
    print("2. Export CSV")
    print("3. Export Excel")

    choice = input("Choose an option: ")
    if choice == "1":
        show_attendance()
    elif choice == "2":
        export_csv()
    elif choice == "3":
        export_excel()
    else:
        print("❌ Invalid option")
