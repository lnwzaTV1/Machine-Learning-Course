# Penguin Species Classification using K-Nearest Neighbors (KNN)

โปรเจกต์การจำแนกสายพันธุ์ของนกเพนกวิน (Palmer Penguins Dataset) โดยใช้ขั้นตอนการเรียนรู้ของเครื่องแบบมีผู้สอนด้วยอัลกอริทึม **K-Nearest Neighbors (KNN Classifier)** และทำการปรับค่าไฮเปอร์พารามิเตอร์ $k$ เพื่อค้นหาโมเดลที่มีความแม่นยำสูงสุด

---

## ภาพรวมโปรเจกต์ (Project Overview)

โปรเจกต์นี้มีเป้าหมายในการจำแนกสายพันธุ์ของนกเพนกวิน 3 สายพันธุ์ (*Adelie*, *Chinstrap*, *Gentoo*) โดยใช้ลักษณะทางกายภาพ 4 ด้านเป็นฟีเจอร์หลัก (Features)

###  ฟีเจอร์ที่ใช้ในการวิเคราะห์
1. **Culmen Length (mm)**: ความยาวของจะงอยปากบน
2. **Culmen Depth (mm)**: ความหนา/ความลึกของจะงอยปากบน
3. **Flipper Length (mm)**: ความยาวของครีบปีก
4. **Body Mass (g)**: มวลน้ำหนักตัว (กรัม)

**เป้าหมาย (Target):** `Species` (สายพันธุ์ของเพนกวิน)

---

##  ขั้นตอนการทำงานของโค้ด (Workflow & Methodology)

1. **Data Cleaning & Preparation**:
   - เลือกเฉพาะคอลัมน์ฟีเจอร์ที่เกี่ยวข้องและคอลัมน์เป้าหมาย
   - ลบแถวที่มีค่าสูญหาย (Missing values) ด้วย `.dropna()` เพื่อให้โมเดลได้รับข้อมูลที่สมบูรณ์

2. **Stratified Train-Test Split**:
   - แบ่งชุดข้อมูลออกเป็น **Train (70%)** และ **Test (30%)**
   - ใช้ `stratify=y` เพื่อคงสัดส่วนของแต่ละสายพันธุ์ให้สมดุลเท่ากันทั้งในชุด Train และ Test
   - กำหนด `random_state=1` เพื่อให้ผลลัพธ์สามารถทำซ้ำได้ (Reproducibility)

3. **Feature Scaling (Standardization)**:
   - เนื่องจาก KNN ใช้ระยะทาง (Euclidean Distance) ในการตัดสิน การที่ตัวแปร `Body Mass (g)` มีหลักพัน ในขณะที่ `Culmen Depth (mm)` มีหลักสิบ จะทำให้ตัวแปรที่มีสเกลใหญ่ส่งผลต่อโมเดลมากเกินไป
   - ใช้ `StandardScaler` เพื่อปรับค่าเฉลี่ย ($\mu = 0$) และส่วนเบี่ยงเบนมาตรฐาน ($\sigma = 1$)
   - ทำ `.fit_transform()` บน Train Data และ `.transform()` บน Test Data เพื่อป้องกันปัญหา **Data Leakage**

4. **Hyperparameter Tuning ($k$-values)**:
   - ทำการทดสอบค่าเพื่อนบ้าน $k \in [3, 5, 7]$
   - วัดผลความแม่นยำ (Accuracy Score) บนชุด Test Data และคัดเลือกค่า $k$ ที่ให้ผลลัพธ์ดีที่สุด

---

##  ข้อกำหนดและไลบรารีที่ใช้ (Requirements)

ติดตั้งไลบรารีที่จำเป็นผ่าน pip:

```bash
pip install pandas scikit-learn
```

---

##  โค้ดโปรแกรม (Source Code)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. โหลดข้อมูล
df = pd.read_csv('penguins_lter.csv')

# 2. กำหนด Features และ Target
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Species'

# 3. จัดการข้อมูลสูญหายและแยก X, y
df_clean = df[features + [target]].dropna()
X = df_clean[features]
y = df_clean[target]

# 4. แบ่งข้อมูล Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# 5. ปรับสเกลข้อมูล (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. เปรียบเทียบค่า k และประเมินผลโมเดล
k_values = [3, 5, 7]
results = {}

print("--- KNN Evaluation Results ---")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[k] = acc
    print(f"k = {k}: Accuracy = {acc * 100:.2f}%")

best_k = max(results, key=results.get)
print(f"\nBest k value: {best_k} with Accuracy: {results[best_k] * 100:.2f}%")
```

---

## การนำไปต่อยอด (Future Improvements)

- นำ **$K$-Fold Cross Validation** หรือ `GridSearchCV` มาใช้เพื่อค้นหาค่าไฮเปอร์พารามิเตอร์ที่ครอบคลุมยิ่งขึ้น
- เพิ่มการประเมินผลด้วย **Confusion Matrix** และ **Classification Report** (Precision, Recall, F1-Score)
- เปรียบเทียบประสิทธิภาพร่วมกับอัลกอริทึมอื่น เช่น Random Forest หรือ Support Vector Machines (SVM)
