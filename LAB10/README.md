# YOLO Penguin Detection

โปรเจกต์นี้ใช้ YOLOv8 ตรวจจับและจำแนกเพนกวิน 3 ชนิดจากภาพ วิดีโอ หรือกล้องเว็บแคม โดยใช้ Dataset รูปภาพที่มีกรอบวัตถุและ class labels ในรูปแบบ YOLO

> `penguins_lter.csv` เป็นข้อมูลตาราง ไม่ได้ใช้ฝึกโมเดลตรวจจับภาพ

## ระบบทำงานอย่างไร

1. `dataset/data.yaml` ระบุ path ของชุดข้อมูลและชื่อคลาส
2. `train.py` โหลด pretrained model `yolov8n.pt` แล้วฝึกการทดลอง 5 แบบ โดยเลือก CUDA GPU อัตโนมัติเมื่อ PyTorch มองเห็น GPU ได้
3. Ultralytics บันทึก weights, metrics และกราฟแยกตามการทดลองไว้ใน `runs/detect/`
4. `evaluate.py` ประเมิน `best.pt` ของแต่ละการทดลองบน test set และสร้าง `runs/test_metrics.csv`
5. `compare_results.py` อ่าน CSV ดังกล่าวและสร้างกราฟเปรียบเทียบผลการทดลอง
6. `predict_image.py`, `predict_video.py` และ `predict_camera.py` โหลดโมเดลแล้วตรวจจับวัตถุจาก input แต่ละชนิด

โมเดลตรวจจับกรอบของเพนกวินและทำนายคลาส:

- Chinstrap Penguin
- Emperor Penguin
- Southern Rockhopper Penguin

## สิ่งที่ต้องมี

- Python 3.10 ขึ้นไป
- แพ็กเกจ `ultralytics`, `torch`, `PyYAML`, `pandas`, `matplotlib` และ `opencv-python`
- Dataset รูปภาพพร้อม YOLO labels
- สำหรับฝึกด้วย GPU ต้องติดตั้ง PyTorch รุ่น CUDA ที่รองรับ GPU และ driver ของเครื่อง

ติดตั้งแพ็กเกจใน virtual environment ที่ต้องการใช้:

```powershell
python -m pip install ultralytics PyYAML pandas matplotlib opencv-python
```

ตรวจว่า PyTorch มองเห็น GPU หรือไม่:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

หากผลเป็น `False` การฝึกจะใช้ CPU ซึ่งช้ากว่า GPU

## โครงสร้าง Dataset

วางไฟล์และโฟลเดอร์ไว้ใต้ `LAB10/dataset/`:

```text
dataset/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

ไฟล์ `data.yaml` ที่ใช้ในโปรเจกต์กำหนด `train`, `val` และ `test` เป็น `train/images`, `valid/images` และ `test/images` ตามลำดับ โฟลเดอร์ `labels` ต้องมี annotation แบบ YOLO ที่ตรงกับชื่อไฟล์ภาพ และลำดับ class ID ต้องตรงกับ `names` ใน YAML

## วิธีใช้งาน

รันคำสั่งจากโฟลเดอร์หลัก `ML-CPE` ใน PowerShell

ก่อนรัน ให้ activate virtual environment ที่ติดตั้ง dependencies ไว้ หากต้องการใช้ GPU ให้เลือก environment ที่ติดตั้ง PyTorch รุ่น CUDA และตรวจสอบว่า `torch.cuda.is_available()` เป็น `True`

### ฝึกโมเดล

```powershell
python .\LAB10\train.py
```

การทดลองที่กำหนดใน `train.py`:

| ชื่อการทดลอง | Epochs | Image size | Batch |
|---|---:|---:|---:|
| `exp1_e10_img640_b8` | 10 | 640 | 8 |
| `exp2_e30_img640_b8` | 30 | 640 | 8 |
| `exp3_e50_img640_b8` | 50 | 640 | 8 |
| `exp4_e50_img320_b8` | 50 | 320 | 8 |
| `exp5_e50_img640_b4` | 50 | 640 | 4 |

การทดลองแต่ละชุดเริ่มจาก `yolov8n.pt` และรันต่อกันตามลำดับ การเรียก `train.py` ใหม่จะเริ่มลูปการทดลองใหม่ ไม่ได้ resume อัตโนมัติ หากต้องการฝึกต่อจาก checkpoint ให้ใช้ `last.pt` กับการตั้งค่า resume ของ Ultralytics

### ประเมินผลบน Test set

```powershell
python .\LAB10\evaluate.py
```

โปรแกรมประเมินเฉพาะการทดลองที่มี `best.pt` และข้ามชุดที่ยังไม่มี weights ผลลัพธ์คือ `LAB10/runs/test_metrics.csv`

### สร้างกราฟเปรียบเทียบ

หลังจาก `evaluate.py` สร้าง CSV แล้ว ให้รัน:

```powershell
python .\LAB10\compare_results.py
```

กราฟจะถูกบันทึกที่ `LAB10/runs/performance_comparison.png`

### ทดสอบภาพ

วางภาพชื่อ `sample.jpg` ในโฟลเดอร์ `LAB10` แล้วรัน:

```powershell
python .\LAB10\predict_image.py
```

ภาพพร้อมกรอบที่ทำนายจะอยู่ใน `LAB10/runs/predict/image/`

### ทดสอบวิดีโอ

วางวิดีโอชื่อ `sample.mp4` ในโฟลเดอร์ `LAB10` แล้วรัน:

```powershell
python .\LAB10\predict_video.py
```

ผลลัพธ์จะอยู่ใน `LAB10/runs/predict/video/`

### ทดสอบกล้องเว็บแคม

```powershell
python .\LAB10\predict_camera.py
```

กด `q` ในหน้าต่างกล้องเพื่อออก

สคริปต์ทดสอบภาพ วิดีโอ และกล้องปัจจุบันใช้ `best.pt` จาก `exp3_e50_img640_b8` หากต้องการใช้โมเดลชุดอื่น ให้แก้ค่า `WEIGHTS` ในสคริปต์ให้ชี้ไปยังไฟล์ `best.pt` ที่ต้องการ

## ไฟล์ผลลัพธ์

- `runs/detect/<experiment>/weights/best.pt`: weights ที่ดีที่สุดของการทดลอง
- `runs/detect/<experiment>/weights/last.pt`: checkpoint ล่าสุด ใช้สำหรับ resume
- `runs/detect/<experiment>/results.csv`: loss และ metrics ราย epoch
- `runs/detect/<experiment>/results.png`: กราฟ training/validation ที่สร้างโดย Ultralytics
- `runs/test_metrics.csv`: metrics บน test set
- `runs/performance_comparison.png`: กราฟเปรียบเทียบผล test
- `runs/predict/`: ผลการทำนายภาพและวิดีโอ

## สถานะผลการฝึกที่มีอยู่

จากไฟล์ผลลัพธ์ในโฟลเดอร์ `runs/detect/`:

- การทดลองที่ 1, 2 และ 3 ฝึกครบตามจำนวน epoch ที่กำหนด
- การทดลองที่ 4 หยุดที่ epoch 26 จาก 50; มี `best.pt`, `last.pt` และกราฟ `training_curves.png`
- การทดลองที่ 5 ยังไม่มีผลการฝึก

ดังนั้นการประเมินหรือกราฟเปรียบเทียบจะแสดงเฉพาะชุดที่มี `best.pt` และอาจรวมผลของการทดลองที่ 4 ซึ่งยังฝึกไม่ครบ
