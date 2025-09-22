import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.db_utils import fetch_all_attendance

def generate_daily_report(output_file: str = "./attendance_report.csv") -> None:
    """
    Generate a daily attendance report from raw logs in the database.

    Args:
        output_file (str): Path to save the generated CSV report.
    """
    # Lấy toàn bộ log từ DB
    records = fetch_all_attendance()
    if not records:
        print("⚠️ No attendance records found.")
        return

    # Tạo DataFrame từ records
    df = pd.DataFrame(records, columns=["ID", "Name", "Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = df["Timestamp"].dt.date

    report = []
    # Gom nhóm theo nhân viên + ngày
    for (name, date), group in df.groupby(["Name", "Date"]):
        check_in = group["Timestamp"].min()
        check_out = group["Timestamp"].max()
        total_time = check_out - check_in
        report.append([name, date, check_in.time(), check_out.time(), total_time])

    # Tạo DataFrame báo cáo
    report_df = pd.DataFrame(
        report,
        columns=["Name", "Date", "Check-in", "Check-out", "Total Time"]
    )

    # Xuất ra CSV
    report_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Daily attendance report exported to {output_file}")

def generate_daily_report_excel(output_file: str = "./attendance_report.xlsx") -> None:
    """
    Generate a daily attendance report and export to Excel.
    """
    records = fetch_all_attendance()
    if not records:
        print("⚠️ No attendance records found.")
        return

    df = pd.DataFrame(records, columns=["ID", "Name", "Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = df["Timestamp"].dt.date

    report = []
    for (name, date), group in df.groupby(["Name", "Date"]):
        check_in = group["Timestamp"].min()
        check_out = group["Timestamp"].max()
        total_time = check_out - check_in
        report.append([name, date, check_in.time(), check_out.time(), total_time])

    report_df = pd.DataFrame(
        report,
        columns=["Name", "Date", "Check-in", "Check-out", "Total Time"]
    )

    report_df.to_excel(output_file, index=False)
    print(f"✅ Daily attendance report exported to {output_file}")

if __name__ == "__main__":
    print("1. Export daily report to CSV")
    print("2. Export daily report to Excel")
    choice = input("Choose an option: ")

    if choice == "1":
        generate_daily_report()
    elif choice == "2":
        generate_daily_report_excel()
    else:
        print("❌ Invalid option")
