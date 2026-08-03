#  LAB 3: Regression & Classification (Palmer Penguins)

##  1. Data Preprocessing & Cleaning

###  Feature Selection & Cleaning
* **คัดเลือกตัวแปร (5 Features):** `Culmen Length`, `Culmen Depth`, `Flipper Length`, `Body Mass` และ `Sex`
* **การจัดการข้อมูลสูญหาย:** ลบแถวที่มีค่าว่าง (`NaN`) ออกทั้งหมดด้วย `dropna()`
* **การกรองข้อมูลขยะ:** คัดเอาเฉพาะแถวที่มีเพศระบุชัดเจนเป็น `'MALE'` หรือ `'FEMALE'` เท่านั้น
* **Label Encoding:** แปลงตัวแปรกลุ่มเป็นตัวเลขเพื่อใช้ในการคำนวณ (`MALE = 1`, `FEMALE = 0`)

>  **ผลลัพธ์:** ได้ชุดข้อมูลที่สะอาดพร้อมนำไปเทรนโมเดลจำนวน **333 แถว (6 คอลัมน์)**

---

##  2. LAB 1: Regression (Body Mass Prediction)

ทดลองสร้างแบบจำลองเพื่อทำนาย **น้ำหนักตัวนกเพนกวิน (`Body Mass`)** โดยแบ่งข้อมูลแบบ **Train 80% / Test 20%**:

| ประเภทโมเดล | ตัวแปรที่ใช้ (Features) | R² Score | RMSE (กรัม) |
| :--- | :--- | :---: | :---: |
| **Simple Linear Regression** | `Flipper Length` (ตัวเดียว) | `0.7938` | `360.40` |
| **Multiple Linear Regression** | `Culmen Length`, `Culmen Depth`, `Flipper Length` | **`0.7981`** | **`356.65`** |

 **สรุปผล Regression:**  
การใช้หลายตัวแปรร่วมกัน (**Multiple Linear Regression**) ให้ค่า $R^2$ สูงกว่าและค่าความคลาดเคลื่อน (RMSE) ต่ำกว่า การใช้ตัวแปรเดียวเพียงอย่างเดียว

---

##  3. LAB 2 & 3: PCA + Classification (Gender Prediction)

###  Feature Scaling & PCA
1. **Standardization:** ปรับสเกลข้อมูลเชิงตัวเลขทั้ง 4 ตัวแปรด้วย `StandardScaler`
2. **PCA Reduction:** บีบอัดมิติข้อมูลจาก **4 Features เหลือเพียง 2 Components (PCA 1, PCA 2)** เพื่อลดความซับซ้อนและช่วยให้แสดงผลบนพิกัด 2D ได้

---

###  Logistic Regression Results
* **Accuracy (ความแม่นยำภาพรวม):** **`80.60%`** (`0.8059`)

####  Classification Metrics Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **FEMALE (0)** | `0.85` | `0.78` | `0.82` | 37 |
| **MALE (1)** | `0.76` | `0.83` | `0.79` | 30 |

---

###  Visualization Output
* **Decision Boundary (กราฟซ้าย):** แสดงเส้นขอบเขตการตัดสินใจจำแนกเพศบนพิกัด PCA 2D (ฝั่งสีฟ้า = เพศหญิง / ฝั่งสีแดง = เพศชาย)
* **Confusion Matrix (กราฟขวา):**
  * **FEMALE:** ทำนายถูกต้อง **29 ตัว** (ทำนายผิดเป็นชาย 8 ตัว)
  * **MALE:** ทำนายถูกต้อง **25 ตัว** (ทำนายผิดเป็นหญิง 5 ตัว)

---

##  Tech Stack
* **Language:** Python 3
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
