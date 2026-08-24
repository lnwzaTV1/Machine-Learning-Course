```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
- `pandas` และ `numpy`: จัดการโครงสร้างข้อมูลแบบตาราง (DataFrame) และการคำนวณเชิงตัวเลข
- `matplotlib.pyplot` และ `seaborn`: ใช้สำหรับการสร้างกราฟและวิเคราะห์การกระจายตัวของข้อมูล
- `sklearn.model_selection.train_test_split`: ฟังก์ชันแบ่งข้อมูลเป็น Train Set และ Test Set
- `sklearn.preprocessing.StandardScaler`: เครื่องมือปรับสเกลข้อมูลให้อยู่ในมาตรฐานเดียวกัน (Mean = 0, Std = 1)
- `sklearn.svm.SVC`: คลาสสำหรับสร้างโมเดล Support Vector Classifier
- `sklearn.metrics`: โมดูลสำหรับวัดประสิทธิภาพโมเดล เช่น ความถูกต้อง (Accuracy) และตาราง Classification Report

### ขั้นตอนที่ 2: การโหลดและทำความสะอาดข้อมูล (Data Loading & Cleaning)
```python
# 1. โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('penguins_lter.csv')

# 2. กำหนดฟีเจอร์ที่ต้องการใช้งาน และลบแถวที่มีค่าสูญหาย (Missing Values / NaN)
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
df_clean = df.dropna(subset=features + ['Species']).copy()

# 3. ตัดทอนชื่อสายพันธุ์ให้กระชับ (ดึงเฉพาะคำแรก เช่น 'Adelie', 'Gentoo', 'Chinstrap')
df_clean['Species_Short'] = df_clean['Species'].apply(lambda x: x.split()[0])

# 4. แยกตัวแปรต้น (X) และตัวแปรตาม (y)
X = df_clean[features]
y = df_clean['Species_Short']

print(f"จำนวนข้อมูลทั้งหมดหลังทำความสะอาด: {df_clean.shape[0]} ตัวอย่าง")
print(f"การกระจายตัวของคลาส:
{y.value_counts()}
")

### ขั้นตอนที่ 3: การแบ่งชุดข้อมูลสำหรับการฝึกสอนและทดสอบ (Train-Test Split)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```
- `test_size=0.25`: กำหนดให้แบ่งข้อมูลสำหรับทดสอบ (Test Set) 25% และสำหรับฝึกสอน (Train Set) 75%
- `random_state=42`: ล็อกค่า Seed สำหรับสุ่ม เพื่อให้ผลการรันซ้ำได้ผลลัพธ์เดิมทุกครั้ง
- `stratify=y`: รักษาสัดส่วนของแต่ละสายพันธุ์ใน Train Set และ Test Set ให้เท่ากับสัดส่วนในชุดข้อมูลจริง ป้องกันปัญหา Imbalanced Data

---


### ขั้นตอนที่ 4: การปรับสเกลข้อมูล (Feature Scaling)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### ขั้นตอนที่ 5: การฝึกโมเดล SVM และเปรียบเทียบแต่ละ Kernel
```python
kernels = ['linear', 'poly', 'rbf']
results = {}

print("="*60)
print("สรุปผลการทดสอบ SVM แต่ละ Kernel Function")
print("="*60)

for k in kernels:
    # สร้างและฝึกโมเดล SVM ตาม Kernel ที่กำหนด
    model = SVC(kernel=k, C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # ทำนายผลบน Test Set และคำนวณ Accuracy
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # จัดเก็บผลลัพธ์
    results[k] = {
        'model': model,
        'accuracy': acc,
        'predictions': y_pred
    }
    
    print(f"Kernel: {k.upper():<10} | Accuracy: {acc * 100:.2f}% ({acc:.4f})")
    print(f"จำนวน Support Vectors: {model.n_support_} (ต่อแต่ละคลาส)")
    print("-" * 60)