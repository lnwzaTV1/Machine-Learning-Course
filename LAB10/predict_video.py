from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "detect" / "exp3_e50_img640_b8" / "weights" / "best.pt"
SOURCE = ROOT / "sample.mp4"  # เปลี่ยนเป็นชื่อวิดีโอของคุณ

if not WEIGHTS.exists():
    raise FileNotFoundError("ไม่พบ best.pt กรุณาฝึกโมเดลก่อน")
if not SOURCE.exists():
    raise FileNotFoundError(f"ไม่พบวิดีโอ {SOURCE}")

model = YOLO(str(WEIGHTS))
model.predict(source=str(SOURCE), conf=0.25, save=True, project=str(ROOT / "runs" / "predict"), name="video")
print("ผลลัพธ์อยู่ใน runs/predict/video/")
