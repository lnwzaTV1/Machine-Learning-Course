from pathlib import Path
import csv
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data.yaml"
RUNS = ROOT / "runs" / "detect"
EXPERIMENTS = [
    "exp1_e10_img640_b8",
    "exp2_e30_img640_b8",
    "exp3_e50_img640_b8",
    "exp4_e50_img320_b8",
    "exp5_e50_img640_b4",
]

def main():
    if not DATA.exists():
        raise FileNotFoundError("ไม่พบ dataset/data.yaml")
    rows = []
    for name in EXPERIMENTS:
        weights = RUNS / name / "weights" / "best.pt"
        if not weights.exists():
            print("ข้าม ยังไม่มีโมเดล:", weights)
            continue
        model = YOLO(str(weights))
        metrics = model.val(data=str(DATA), split="test", plots=True)
        rows.append({
            "experiment": name,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        })
    out = ROOT / "runs" / "test_metrics.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("บันทึกผลแล้ว:", out)
        for row in rows:
            print(row)
    else:
        print("ยังไม่มีโมเดลที่ฝึกเสร็จ กรุณารัน train.py ก่อน")

if __name__ == "__main__":
    main()
