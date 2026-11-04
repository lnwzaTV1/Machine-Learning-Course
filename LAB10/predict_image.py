from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "detect" / "exp3_e50_img640_b8" / "weights" / "best.pt"
SOURCE = ROOT / "sample.jpg"  # เปลี่ยนเป็นชื่อภาพที่ต้องการทดสอบ

if not WEIGHTS.exists():
    raise FileNotFoundError("ไม่พบ best.pt กรุณาฝึกโมเดลก่อน")
if not SOURCE.exists():
    raise FileNotFoundError(f"ไม่พบภาพ {SOURCE} กรุณาวาง sample.jpg ไว้ในโฟลเดอร์โปรเจกต์")

model = YOLO(str(WEIGHTS))
model.predict(source=str(SOURCE), conf=0.25, save=True, project=str(ROOT / "runs" / "predict"), name="image")
print("ผลลัพธ์อยู่ใน runs/predict/image/")
