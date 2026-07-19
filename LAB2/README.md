# LAB02: Data Preprocessing

This lab is all about preparing and cleaning a raw dataset before using it to train Machine Learning models.

## 1. Dataset Exploration
* **Dataset Used:** Student Performance Dataset from Kaggle.
* **Data Shape:** 395 rows and 33 columns.
* **Class Distribution (Gender):** 208 Females (F) and 187 Males (M).
* **Missing & Duplicate Check:** Checked everything! No missing values (NaN) and zero duplicate records found.

## 2. Data Cleaning
* Used `drop_duplicates()` to ensure there are no repeated rows.
* Converted the `age` column to `int64` format for clean calculations.
* **Final Exam Grade (G3) Statistics:**
  * Mean (Average): 10.42
  * Median (Middle value): 11.00

## 3. Feature Engineering
* **Label Encoding:** Converted binary text columns (`sex` and `schoolsup`) into numbers (0 and 1) so the model can understand them.
* **One-Hot Encoding:** Split the `Mjob` column (Mother's job) into multiple sub-columns (like `Mjob_at_home`, `Mjob_health`, `Mjob_services`) using 0 and 1.

---
### Dataset Credit
* **Source:** Kaggle Datasets
* **URL:** [Student Performance Data](https://www.kaggle.com/datasets/devansodariya/student-performance-data)