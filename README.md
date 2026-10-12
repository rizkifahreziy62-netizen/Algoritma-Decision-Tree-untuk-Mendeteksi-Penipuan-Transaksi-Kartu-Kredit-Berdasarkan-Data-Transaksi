
# Analisis Metrik Evaluasi Model Machine Learning
## MAE, MSE, dan R² pada Dataset Credit Card Fraud Detection

### 📌 Deskripsi Proyek
Proyek ini membahas metrik evaluasi model Machine Learning, khususnya Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Coefficient of Determination (R²). Tujuannya adalah memahami fungsi masing-masing metrik, menentukan nilai idealnya, serta membuktikannya berdasarkan referensi jurnal ilmiah dan dokumentasi teknis.

Dataset yang digunakan adalah **Credit Card Fraud Detection Dataset** dari Kaggle.

### 🎯 Tujuan
- Memahami pengertian dan fungsi MAE, MSE, dan R².
- Mengetahui nilai ideal setiap metrik evaluasi.
- Membuktikan interpretasi metrik menggunakan referensi ilmiah.
- Memahami penerapan metrik evaluasi dalam Machine Learning.
- Menentukan metrik yang sesuai dengan karakteristik dataset.

### 📊 Metrik Evaluasi

| Metrik | Nilai Ideal | Fungsi |
|---|---|---|
| MAE | Mendekati 0 | Mengukur rata-rata kesalahan absolut prediksi. |
| MSE | Mendekati 0 | Mengukur rata-rata kuadrat kesalahan prediksi. |
| R² | Mendekati 1 | Mengukur kemampuan model menjelaskan variasi target. |

#### 1. Mean Absolute Error (MAE)
MAE menghitung rata-rata selisih absolut antara nilai aktual dan nilai prediksi.

Rumus:

MAE = (1/n) × Σ |yᵢ - ŷᵢ|

Semakin kecil nilai MAE, semakin kecil rata-rata kesalahan prediksi.

#### 2. Mean Squared Error (MSE)
MSE menghitung rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi.

Rumus:

MSE = (1/n) × Σ (yᵢ - ŷᵢ)²

Semakin kecil nilai MSE, semakin kecil kesalahan kuadrat rata-rata. MSE memberikan penalti lebih besar terhadap kesalahan yang besar.

#### 3. Coefficient of Determination (R²)
R² digunakan untuk mengukur seberapa baik model menjelaskan variasi nilai target.

Rumus:

R² = 1 - (SSres / SStot)

Interpretasi:
- R² = 1: prediksi sempurna pada data evaluasi.
- R² = 0: performa setara dengan baseline prediksi rata-rata dalam definisi standar.
- R² < 0: model dapat lebih buruk daripada baseline tersebut.

### 📁 Dataset
**Nama:** Credit Card Fraud Detection Dataset

**Sumber:** Kaggle

**Link:** https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset

Dataset ini digunakan untuk mempelajari karakteristik transaksi kartu kredit dan mendeteksi transaksi yang terindikasi penipuan.

### 🛠️ Tools dan Teknologi
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### 📚 Referensi Ilmiah

1. Chicco, D., Warrens, M. J., & Jurman, G. (2021). *The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation.* PeerJ Computer Science, 7, e623.  
   https://doi.org/10.7717/peerj-cs.623

2. Scikit-learn Developers. *Model evaluation: quantifying the quality of predictions.*  
   https://scikit-learn.org/stable/modules/model_evaluation.html

3. Scikit-learn Developers. *Mean Absolute Error.*  
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

4. Scikit-learn Developers. *Mean Squared Error.*  
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

### ⚠️ Catatan
Dataset dengan target `is_fraud` umumnya merupakan permasalahan klasifikasi karena targetnya menunjukkan kelas transaksi penipuan atau bukan penipuan.

Untuk mengevaluasi model klasifikasi, metrik yang umumnya lebih sesuai adalah:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

MAE, MSE, dan R² dibahas dalam konteks evaluasi regresi. Penggunaan metrik tersebut pada target biner perlu disesuaikan dengan tujuan analisis.

### 📝 Kesimpulan
MAE dan MSE memiliki nilai ideal mendekati 0 karena keduanya mengukur kesalahan prediksi. Sebaliknya, R² memiliki nilai ideal mendekati 1 karena menunjukkan kemampuan model menjelaskan variasi target.

Pemilihan metrik evaluasi harus mempertimbangkan jenis permasalahan, karakteristik dataset, dan tujuan penelitian.

---
**Proyek Machine Learning — Evaluasi Metrik Model**
