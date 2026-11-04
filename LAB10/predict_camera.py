from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "detect" / "exp3_e50_img640_b8" / "weights" / "best.pt"

if not WEIGHTS.exists():
    raise FileNotFoundError("ไม่พบ best.pt กรุณาฝึกโมเดลก่อน")

model = YOLO(str(WEIGHTS))
# กด q ที่หน้าต่างกล้องเพื่อออก
model.predict(source=0, show=True, conf=0.25)
