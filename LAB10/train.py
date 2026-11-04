from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "data.yaml"
PROJECT = ROOT / "runs" / "detect"

def check_dataset():
    if not DATA.exists():
        raise FileNotFoundError(
            f"ไม่พบ {DATA}\n"
            "กรุณาดาวน์โหลด Dataset แบบ YOLOv8 จาก Roboflow "
            "แล้ววางโฟลเดอร์ train/valid/test และ data.yaml ไว้ใน dataset/"
        )
    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        if split not in cfg:
            raise ValueError(f"data.yaml ต้องมี key '{split}'")
    print("Dataset YAML:", cfg)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

def main():
    check_dataset()
    # ทุกการทดลองเริ่มจาก pretrained weights เดียวกัน
    experiments = [
        {"name": "exp1_e10_img640_b8", "epochs": 10, "imgsz": 640, "batch": 8},
        {"name": "exp2_e30_img640_b8", "epochs": 30, "imgsz": 640, "batch": 8},
        {"name": "exp3_e50_img640_b8", "epochs": 50, "imgsz": 640, "batch": 8},
        {"name": "exp4_e50_img320_b8", "epochs": 50, "imgsz": 320, "batch": 8},
        {"name": "exp5_e50_img640_b4", "epochs": 50, "imgsz": 640, "batch": 4},
    ]

    device = 0 if torch.cuda.is_available() else "cpu"
    for exp in experiments:
        print(f"\n===== START {exp['name']} =====")
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(DATA),
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            device=device,
            project=str(PROJECT),
            name=exp["name"],
            exist_ok=True,
            workers=0,       # ลดปัญหา multiprocessing บน Windows
            patience=20,
            plots=True,
            save=True,
            pretrained=True
        )
        print(f"===== FINISHED {exp['name']} =====")

if __name__ == "__main__":
    main()
