# Laporan Proyek Machine Learning - Rahayu Kartika Sari

## Domain proyek

![Wine Turbine](https://www.solarfeeds.com/mag/wp-content/uploads/2019/10/picspree-1256331.jpg)

Energi terbarukan, khususnya dari kincir angin, memainkan peran penting dalam transisi menuju sumber energi berkelanjutan [1](https://www.tandfonline.com/doi/abs/10.1016/j.clipol.2003.10.010). Dengan semakin meningkatnya kebutuhan energi dan kekhawatiran tentang perubahan iklim, penting untuk memanfaatkan teknologi yang ada untuk memprediksi dan mengoptimalkan hasil energi dari sumber ini.

Proyek ini bertujuan untuk menganalisis dan memprediksi energi yang dihasilkan oleh kincir angin menggunakan dataset yang berisi data SCADA (Supervisory Control and Data Acquisition) dari kincir angin di Turki pada tahun 2018.

### Mengapa dan Bagaimana Masalah Ini Dapat Dipecahkan

Dalam konteks dunia yang semakin menuntut penggunaan energi bersih, pemahaman yang lebih baik tentang variabel-variabel yang mempengaruhi output energi kincir angin dapat memberikan wawasan yang berharga bagi operator dan pengembang proyek energi terbarukan. Dengan menggunakan algoritma _Machine Learning_, proyek ini dapat membantu dalam menghasilkan model yang lebih akurat untuk memprediksi produksi energi berdasarkan faktor-faktor seperti kecepatan angin, suhu, dan kondisi lingkungan lainnya [2](https://www.mdpi.com/1996-1073/14/1/125).

## Business Understanding

### Problem Statements

- Hasil energi dari kincir angin sangat dipengaruhi oleh faktor cuaca dan lingkungan yang berfluktuasi, membuatnya sulit untuk memprediksi dengan akurat hanya berdasarkan pengamatan manual.
- Banyak data yang terkait dengan kinerja kincir angin tidak terstruktur dan tersebar, menyulitkan analisis dan pengambilan keputusan yang cepat dan tepat dari operator.
- Proses pengambilan keputusan terkait kapabilitas produksi seringkali memakan waktu dan memerlukan alat analitik yang dapat mengolah data secara efisien.

### Goals

- Mengembangkan model pembelajaran mesin untuk memprediksi energi yang dihasilkan oleh kincir angin dengan akurasi tinggi, berdasarkan dataset yang tersedia, sehingga memungkinkan pengambilan keputusan yang lebih tepat dan cepat.
- Memanfaatkan data historis untuk mengidentifikasi pola dan hubungan antara variabel seperti kecepatan angin, sudut bilah kincir, dan kondisi lingkungan lainnya, guna meningkatkan hasil prediksi.
- Menyediakan alat prediksi yang dapat diakses oleh pengelola proyek energi untuk meningkatkan efisiensi operasional dan penjadwalan pemeliharaan.

### Solution statements

- Melakukan percobaan dengan menggunakan beberapa algoritma machine learning, termasuk ARIMA, RNN, LSTM, Bi-LSTM, dan GRU untuk membandingkan akurasi dan efektivitas masing-masing model dalam memprediksi energi yang dihasilkan berdasarkan data waktu yang tersedia. Pemilihan algoritma ini bertujuan untuk memanfaatkan kelebihan masing-masing dalam menangkap pola dalam dataset yang berskala waktu.
- Melakukan eksplorasi data secara mendalam yang mencakup univariate analysis untuk memahami distribusi tiap fitur, multivariate analysis untuk mengidentifikasi interaksi antar fitur, serta correlation analysis untuk mengetahui hubungan yang signifikan antara fitur-fitur kunci seperti kecepatan angin, suhu, dan output daya. Proses ini bertujuan untuk mendapatkan wawasan yang lebih dalam mengenai faktor-faktor yang mempengaruhi keluaran energi serta untuk melakukan fitur rekayasa yang relevan.
- Mengimplementasikan feature engineering untuk menghasilkan variabel baru dari data yang ada, seperti kecepatan angin rata-rata harian dan tren suhu, agar model dapat menangkap pola yang lebih kompleks dan efektif dalam memprediksi hasil energi.
- Melakukan cross-validation untuk mengevaluasi kinerja model yang dikembangkan, guna memastikan bahwa model tidak hanya akurat pada data pelatihan tetapi juga memiliki kemampuan generalisasi yang baik pada data yang belum pernah dilihat sebelumnya.
- Menerapkan hyperparameter tuning pada model yang terpilih menggunakan teknik seperti Grid Search, untuk mengoptimalkan parameter model sehingga dapat meningkatkan akurasi prediksi secara keseluruhan.

Dengan langkah-langkah ini, proyek bertujuan untuk mencapai efisiensi dan keakuratan tinggi dalam memprediksi energi yang dihasilkan oleh kincir angin, sehingga dapat memberikan informasi yang berguna untuk perencanaan dan manajemen sistem energi terbarukan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [2018 SCADA Data of a Wind Turbine in Turkey](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset/data). Dataset ini berisi informasi operasional dari sebuah kincir angin selama periode tertentu, memperlihatkan berbagai variabel yang berkontribusi pada produksi energi. Data ini terukur pada interval waktu tertentu, mulai dari 1 Januari 2018 hingga 13 Desember 2018.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- Date/Time: Stempel waktu yang mencatat tanggal dan waktu pengukuran dengan interval 10 menit.
- LV ActivePower (kW): Energi yang dihasilkan turbin yang menjadi variabel target
- Wind Speed (m/s): Kecepatan angin yang digunakan turbin untuk menghasilkan listrik
- Wind Direction (degrees): Arah angin yang diukur dalam derajat, membantu dalam memahami bagaimana orientasi kincir angin sehingga berpengaruh terhadap efisiensi.
- Theoretical_Power_Curve (KWh): Prediksi energi yang dihasilkan turbin secara teoritis.

### Tahapan Exploratory Data Analysis (EDA):

Dilakukan proses sebagai berikut:

#### 🧹 Data Cleaning

1.  Missing Data

    - Dilakukan pemeriksaan jumlah nilai yang hilang pada setiap kolom menggunakan `.isna().sum()`.
    - Untuk kolom suhu (`Temperature (°C)`), nilai yang hilang sebelumnya telah diatasi menggunakan metode **interpolasi** saat proses feature engineering.

2.  Outlier Handling
    - Pada kolom `LV ActivePower (kW)` ditemukan nilai **negatif**, yang secara fisik tidak mungkin terjadi karena turbin tidak dapat menghasilkan daya negatif.
    - Jumlah data anomali dihitung, kemudian semua nilai negatif diganti menjadi **0** menggunakan fungsi `.apply()`.
    - Langkah ini penting untuk menjaga validitas data dan mencegah model belajar dari informasi yang salah.

#### 🔧 **Feature Engineering**

Tahapan ini dilakukan untuk memperkaya dataset dengan fitur-fitur tambahan yang relevan, guna meningkatkan kualitas model prediksi energi angin.

1.  **Ekstraksi Waktu**

    - Menambahkan kolom:
    - `Hour`: Jam ke berapa (0–23)
    - `Day`: Hari dalam bulan
    - `Week`: Minggu ke berapa dalam tahun
    - `Month`: Bulan ke berapa
    - Bertujuan menangkap pola harian dan musiman dalam data.

2.  **Identifikasi Musim (Season)**

    - Menentukan musim berdasarkan nilai `Month`:
    - 1 = Winter (Des–Feb)
    - 2 = Spring (Mar–Mei)
    - 3 = Summer (Jun–Agu)
    - 4 = Autumn (Sep–Nov)
    - Fitur ini membantu memahami dampak perubahan musim terhadap kecepatan angin.

3.  **Penentuan Siang/Malam (Day/Night)**

    - Menggunakan library **Astral** untuk menghitung waktu matahari terbit dan terbenam di kota Izmir, Turki.
    - Menambahkan kolom `Day/Night`:
    - 0 = Siang
    - 1 = Malam
    - Bertujuan membedakan perilaku angin antara siang dan malam hari.

4.  **Suhu Udara (Temperature)**
    - Data suhu diperoleh dari **Meteostat API** berdasarkan lokasi dan waktu.
    - Disesuaikan ke interval 10 menit agar sinkron dengan data utama.
    - Fitur `Temperature (°C)` ditambahkan dan nilai yang hilang diisi menggunakan interpolasi.
    - Suhu digunakan karena berpengaruh terhadap densitas udara yang berperan dalam rumus energi kinetik angin.

#### 📊 Data Analysis & Visualisasi

Analisis eksploratif dilakukan untuk memahami pola, distribusi, dan hubungan antar fitur dalam dataset.

1.  Analisis Distribusi dan Korelasi

    - Visualisasi awal menggunakan **Pairplot** dan **Histogram** menunjukkan sebaran dan korelasi antara:

      - `LV ActivePower (kW)`
      - `Wind Speed (m/s)`
      - `Theoretical_Power_Curve (KWh)`
      - `Wind Direction (°)`
      - `Temperature (°C)`

        ![Pairplot](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/pairplot_observed_column.png?raw=true)

        ![Histogram](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/histogram_observes_column.png?raw=true)

    - **Boxplot** digunakan untuk mengidentifikasi outlier, terutama pada kolom `Wind Speed (m/s)`.

2.  Pola Data terhadap Waktu

    - Plot garis menunjukkan pola fluktuasi fitur terhadap waktu.
    - Memberikan wawasan tentang musim, harian, dan perbedaan waktu (siang vs malam) dalam produksi energi.

3.  Distribusi Energi

    - **Siang vs Malam**: Malam hari menghasilkan energi lebih banyak secara akumulatif.
    - **Bulanan**: Produksi energi tertinggi terjadi di bulan **Maret** dan **Agustus**.
    - **Musiman**: Energi paling banyak dihasilkan saat **musim gugur (Autumn)**.
      ![Day/Night Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/accumulation_1.png?raw=true)
      ![Month Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/accumulation_2.png?raw=true)
      ![Season Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/accumulation_3.png?raw=true)

4.  Analisis Kecepatan Angin

    - Hubungan antara `Wind Speed (m/s)` dan:

      - **Theoretical Power** menunjukkan kurva daya ideal turbin.

        ![Wind Speed and Theoretical Power](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/power_1.png?raw=true)

      - **LV ActivePower** menunjukkan performa aktual.

        ![Wind Speed and Actual Power](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/power_2.png?raw=true)

    - Ditemukan:
      - Kecepatan minimum untuk menghasilkan daya ≈ **3.6 m/s**
      - Kecepatan minimum untuk mencapai daya maksimum ≈ **13.8 m/s**

5.  Analisis Arah Angin

    - Arah angin tersebar di semua sudut (0–360°), menunjukkan cakupan penuh oleh turbin.

      ![Wind Direction Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/wind_direction.png?raw=true)

    - Arah angin paling optimal dalam menghasilkan energi adalah sekitar: **30–75°** dan **180–210°**

      ![Wind direction and Power](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/wind_direction_1.png?raw=true)

    - Arah angin ini juga berkorelasi dengan kecepatan angin yang tinggi.

      ![Wind direction and Wind Speed](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/wind_direction_2.png?raw=true)

6.  Korelasi Antar Fitur (Heatmap)
    ![Heatmap](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/heatmap.png?raw=true)

    - Korelasi kuat ditemukan antara `Wind Speed` dan `Theoretical_Power_Curve`, serta `Wind Speed` dengan `LV ActivePower`.
    - Visualisasi ini menegaskan bahwa kecepatan angin adalah fitur paling penting untuk prediksi daya yang dihasilkan.

## 🛠️ Data Preparation

Data Preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan machine learning bersih, konsisten, dan siap untuk dianalisis. Proses ini penting untuk meningkatkan kinerja model dengan mengatasi masalah potensial dalam kualitas data.

### Proses Data Preparation:

1. **Normalisasi**
   Masalah Skala Fitur: Beberapa fitur dalam dataset, seperti Wind Speed dan Temperature, memiliki skala yang sangat berbeda. Misalnya, kecepatan angin dapat berkisar antara 0 hingga 25 m/s, sementara suhu dapat berkisar antara 0 hingga 40 °C. Model yang sensitif terhadap skala fitur (seperti regresi linier atau jaringan saraf) dapat dipengaruhi jika fitur dengan skala yang lebih besar mendominasi proses pelatihan. Oleh karena itu, penting untuk menormalkan fitur numerik.

   **Langkah yang Diambil**:
   Normalisasi dilakukan menggunakan **StandardScaler**, yaitu metode standardisasi yang mengubah distribusi setiap fitur menjadi memiliki nilai rata-rata 0 dan standar deviasi 1, menggunakan rumus:

   ![Standard Scaler Equation](https://journaldev.nyc3.cdn.digitaloceanspaces.com/2020/10/Standardization.png)

   di mana $X$ adalah nilai asli, $\mu$ adalah rata-rata fitur, dan $\sigma$ adalah standar deviasi fitur.

   Alasan Penggunaan:
   Standardisasi digunakan karena metode ini bekerja baik dengan algoritma yang mengasumsikan distribusi data normal atau sensitif terhadap skala, seperti Regresi Linier, SVM, dan Neural Networks. Dengan fitur berada pada skala yang seragam, model dapat mempelajari pola dengan lebih stabil dan efisien.

2. **Splitting Data (Train-Test Split)**

   Pemisahan data menjadi bagian pelatihan dan pengujian penting untuk mencegah overfitting, yaitu kondisi di mana model terlalu “hapal” data pelatihan dan gagal menggeneralisasi ke data baru.

   Langkah yang Diambil:
   Dataset dibagi menjadi:

   - 80% untuk pelatihan `(X_train, y_train)`
   - 20% untuk pengujian `(X_test, y_test)`
     Proses ini dilakukan secara acak dengan `random_state=42` untuk memastikan hasil yang konsisten saat replikasi.

   Manfaat:
   Dengan melakukan pemisahan ini, performa model dapat dievaluasi secara objektif terhadap data yang tidak pernah dilihat sebelumnya, memberikan gambaran lebih realistis tentang akurasi model di dunia nyata.

## 🤖 Modeling

Pemodelan dilakukan untuk memprediksi daya listrik (`LV ActivePower (kW)`) yang dihasilkan turbin angin berdasarkan berbagai fitur seperti kecepatan angin, suhu, arah angin, dan lainnya. Beberapa model regresi digunakan untuk membandingkan performa dan memilih model terbaik.

### 📌 Model yang Digunakan

1. **Gradient Boosting Regressor**
   Gradient Boosting adalah metode ensambel yang membangun model prediksi secara bertahap, di mana setiap model baru berusaha mengoreksi kesalahan dari model sebelumnya. Model ini menggabungkan banyak weak learners (biasanya decision tree) secara bertahap menggunakan gradient descent untuk meminimalkan loss function. Cocok untuk data kompleks yang tidak linear.

   - ✅ Kelebihan:
     - Mampu menangani non-linearitas dan fitur interaksi dengan baik
     - Sering memberikan hasil akurat pada data kompleks
   - ⚠️ Kekurangan:
     - Waktu pelatihan lebih lama
     - Rentan terhadap overfitting jika tidak dituning

2. **Support Vector Regressor (SVR)**
   SVR adalah versi regresi dari Support Vector Machine. SVR mencari fungsi (biasanya non-linear) yang dapat memprediksi nilai target dalam margin toleransi tertentu (epsilon). Tujuannya adalah menemukan hyperplane terbaik yang meminimalkan error sambil menjaga margin maksimum. SVR bekerja baik dengan data kecil atau berdimensi tinggi.

   - ✅ Kelebihan:
     - Baik untuk data berukuran kecil dan linear/non-linear
     - Robust terhadap outlier (dengan kernel dan parameter yang tepat)
   - ⚠️ Kekurangan:
     - Sulit dituning (C, epsilon, kernel)
     - Tidak efisien untuk dataset besar

3. **Random Forest Regressor**
   Random Forest adalah metode ensambel berbasis decision tree. Model ini membangun banyak pohon keputusan pada subset data secara acak dan menggabungkan prediksinya dengan rata-rata. Teknik ini mengurangi overfitting yang sering terjadi pada decision tree tunggal dan meningkatkan akurasi serta stabilitas prediksi.

   - ✅ Kelebihan:
     - Mengurangi overfitting dibandingkan Decision Tree tunggal
     - Stabil dan bekerja baik dengan data yang noisy
   - ⚠️ Kekurangan:
     - Interpretabilitas rendah
     - Lebih lambat dibanding model sederhana

4. **Linear Regression**
   Linear Regression adalah model statistik sederhana yang mengasumsikan hubungan linear antara fitur input dan target output. Model ini mencoba menemukan garis lurus terbaik (dalam ruang multi-dimensi) yang meminimalkan selisih kuadrat antara prediksi dan nilai aktual. Sering digunakan sebagai baseline karena interpretasinya yang mudah.

   - ✅ Kelebihan:
     - Cepat dan mudah diinterpretasikan
     - Cocok sebagai baseline model
   - ⚠️ Kekurangan:
     - Tidak menangani non-linearitas
     - Sensitif terhadap multikolinearitas dan outlier

5. **Extra Trees Regressor** (Extremely Randomized Trees)
   Extra Trees adalah varian dari Random Forest yang melakukan pemisahan (splitting) lebih acak pada tiap node decision tree. Berbeda dari Random Forest yang memilih split terbaik, Extra Trees memilih split secara acak dari subset kandidat. Pendekatan ini membuat model lebih cepat dan membantu mengurangi varians.

   - ✅ Kelebihan:
     - Lebih cepat dari Random Forest
     - Mengurangi varians melalui randomisasi split
   - ⚠️ Kekurangan:
     - Interpretasi hasil rendah
     - Bisa terlalu acak pada data yang kecil

6. **AdaBoost Regressor**
   AdaBoost (Adaptive Boosting) menggabungkan sejumlah model lemah (biasanya decision tree sederhana) secara bertahap. Setiap model fokus pada sampel yang salah diprediksi oleh model sebelumnya. Bobot pada data yang sulit diprediksi dinaikkan agar model selanjutnya lebih fokus padanya. Efektif untuk meningkatkan performa model sederhana.

   - ✅ Kelebihan:
     - Memperbaiki kesalahan model lemah secara iteratif
     - Sederhana dan efektif
   - ⚠️ Kekurangan:
     - Rentan terhadap noise dan outlier
     - Performa bisa turun pada data non-linear jika base learner terlalu lemah

7. **Decision Tree Regressor**
   Decision Tree membagi data ke dalam beberapa simpul berdasarkan nilai fitur tertentu, mengikuti prinsip if-else hingga mencapai prediksi numerik. Model ini mudah dipahami dan divisualisasikan. Namun, model ini cenderung overfit jika tidak diatur dengan baik (pruning, depth, dll.).

   - ✅ Kelebihan:
     - Mudah diinterpretasikan
     - Menangani fitur numerik dan kategorikal tanpa normalisasi
   - ⚠️ Kekurangan:
     - Rentan terhadap overfitting
     - Tidak stabil terhadap perubahan kecil dalam data

8. **XGBoost Regressor**
   XGBoost (Extreme Gradient Boosting) adalah implementasi efisien dan teroptimasi dari gradient boosting. Model ini dilengkapi dengan regularisasi L1 dan L2 untuk menghindari overfitting, serta mendukung paralelisasi dan kontrol penuh terhadap proses boosting. Sangat populer karena kecepatan dan akurasinya yang tinggi.

   - ✅ Kelebihan:
     - Performa tinggi dan efisien
     - Dilengkapi regularisasi untuk menghindari overfitting
   - ⚠️ Kekurangan:
     - Kompleksitas tuning parameter
     - Konsumsi memori relatif tinggi

9. **XGBRF Regressor**
   XGBRF adalah variasi dari XGBoost yang menggabungkan teknik random forest dengan boosting. Alih-alih memfokuskan pada kesalahan model sebelumnya, model ini membangun banyak pohon menggunakan subsampling acak terhadap fitur dan data. Cocok untuk menyeimbangkan akurasi dan stabilitas model.

   - ✅ Kelebihan:
     - Kombinasi akurasi XGBoost dengan prinsip Random Forest
     - Baik untuk ensemble learning
   - ⚠️ Kekurangan:
     - Bisa lambat saat pelatihan
     - Kompleksitas implementasi meningkat

10. **CatBoost Regressor**
    CatBoost adalah algoritma boosting dari Yandex yang dirancang untuk menangani data kategorikal dengan lebih efisien. Model ini secara otomatis menangani fitur kategorikal tanpa perlu encoding eksplisit dan menggunakan teknik ordered boosting untuk menghindari data leakage. Stabil dan sering memberikan hasil kompetitif tanpa banyak tuning.

- ✅ Kelebihan:
  - Performa tinggi, terutama untuk data kategorikal
  - Sedikit tuning, stabil secara default
- ⚠️ Kekurangan:
  - Dokumentasi lebih terbatas dibanding XGBoost
  - Proses pelatihan awal lebih lambat

## ⚙️ Evaluation

Setelah melakukan pelatihan model menggunakan dataset yang sudah dipersiapkan, evaluasi dilakukan untuk menilai kinerja model-model yang telah dibangun. Evaluasi ini menggunakan beberapa metrik yang penting untuk masalah regresi kontinu karena fokus kita adalah memprediksi output daya yang dihasilkan dari kincir angin.

### Metrik Evaluasi yang Digunakan:

1. **Mean Absolute Error (MAE)**

   MAE menghitung rata-rata dari kesalahan absolut antara nilai prediksi dan nilai aktual. Ini memberikan gambaran seberapa banyak kesalahan dalam prediksi dalam satuan yang sama dengan data.

   Rumus:

   ![MAE Equation](https://miro.medium.com/v2/resize:fit:822/1*-fLHZ0gOJg7nTuWVm_kIzw.png)

   Keterangan:

   - $y_i$: Nilai aktual.
   - $\hat{y_i}$: Nilai prediksi.
   - $n$: Jumlah total prediksi.

   **Kelebihan**:

   - Mudah diinterpretasikan dan dihitung.
   - Mengukur kesalahan dalam unit yang sama dengan data.

   **Kekurangan**:

   - Tidak memberikan bobot lebih pada kesalahan yang lebih besar dan bisa jadi kurang sensitif terhadap outlier.

2. **Mean Absolute Percentage Error (MAPE)**

   MAPE mengukur kesalahan rata-rata dalam persentase, memberikan informasi tentang seberapa besar kesalahan model dalam konteks relatif.

   Rumus:

   ![MAPE Equation](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjHL76Vzl5SiPLX9dQm5KimXogVNpMc7KaAx4LMjRmQi1GVpHe8XG-0R55b2_z-Ozd76f5lSaFRe6CyuxWTNqHwwetfdWbSpKI_kpL_tsvnE9xGA5-QO0iwGAMsrGdSVUaV7xJYjcd04FtM/s433/mape.jpg)

   Keterangan:

   - $y_i$: Nilai aktual.
   - $\hat{y_i}$: Nilai prediksi.
   - $n$: Jumlah total prediksi.

   **Kelebihan**:

   - Memberikan informasi yang lebih relevan ketika membandingkan performa model di berbagai skala.
   - Lebih mudah dipahami oleh pengguna non-teknis.

   **Kekurangan**:

   - Tidak dapat dihitung jika nilai aktual $y_i = 0$ karena menghasilkan pembagian oleh nol.

3. **Root Mean Square Error (RMSE)**

   RMSE mengukur seberapa besar kesalahan kuadrat antara nilai prediksi dan nilai aktual, memberikan bobot lebih pada kesalahan yang lebih besar.

   Rumus:

   ![RSME Equation](https://arize.com/wp-content/uploads/2023/08/RMSE-equation.png)

   Keterangan:

   - $y_i$: Nilai aktual.
   - $\hat{y_i}$: Nilai prediksi.
   - $n$: Jumlah total prediksi.

   **Kelebihan**:

   - Lebih sensitif terhadap outlier dibandingkan MAE.
   - Memberikan informasi mengenai kesalahan dalam unit yang sama dengan data.

   **Kekurangan**:

   - Dapat memberikan gambaran yang menyesatkan jika ada outlier yang signifikan.

4. **R-squared (R²)**

   R² adalah metrik yang mengukur proporsi variansi dalam data target yang dapat dijelaskan oleh model. Ini memberikan informasi tentang seberapa baik model memprediksi output berdasarkan input yang diberikan.

   Rumus:

   ![R2 Equation](https://vitalflux.com/wp-content/uploads/2019/07/R-squared-formula-function-of-SSE-and-SST.jpg)

   Keterangan:

   - $y_i$: Nilai aktual.
   - $\hat{y_i}$: Nilai prediksi.
   - $\bar{y}$: Rata-rata dari nilai aktual.
   - $n$: Jumlah total prediksi.

   **Kelebihan**:

   - Memberikan gambaran yang jelas tentang seberapa baik model bekerja.
   - Mudah diinterpretasikan dalam konteks proporsi variansi.

   **Kekurangan**:

   - Hanya dapat digunakan dengan model regresi dan tidak memberikan informasi mengenai kekuatan hubungan antar variabel independen dan dependen.

### 🧪 Hasil Training Model

Setelah proses pelatihan dan evaluasi dilakukan terhadap berbagai algoritma regresi, berikut adalah perbandingan performa masing-masing model berdasarkan empat metrik evaluasi:

| Model                   | R² Score | RMSE   | MAE    | MAPE        |
| ----------------------- | -------- | ------ | ------ | ----------- |
| **CatBoost Regressor**  | 0.9847   | 161.73 | 71.92  | 7.20 × 10¹⁶ |
| **XGBoost Regressor**   | 0.9833   | 168.73 | 72.44  | 7.05 × 10¹⁶ |
| ExtraTrees Regressor    | 0.9782   | 192.76 | 69.95  | 7.04 × 10¹⁶ |
| Random Forest Regressor | 0.9754   | 204.84 | 73.30  | 7.66 × 10¹⁶ |
| Decision Tree Regressor | 0.9569   | 271.19 | 88.62  | 5.87 × 10¹⁶ |
| Gradient Boosting Regr. | 0.9519   | 286.55 | 127.70 | 1.54 × 10¹⁷ |
| XGBRF Regressor         | 0.9464   | 302.54 | 123.74 | 1.48 × 10¹⁷ |
| Linear Regression       | 0.9059   | 400.72 | 207.78 | 3.00 × 10¹⁷ |
| SVR                     | 0.8951   | 423.00 | 165.70 | 2.37 × 10¹⁷ |
| AdaBoost Regressor      | 0.8857   | 441.57 | 273.10 | 2.00 × 10   |

### 🔍 Interpretasi Hasil:

- **CatBoost Regressor** dan **XGBoost Regressor** menunjukkan performa terbaik dengan nilai R² mendekati 1, menunjukkan bahwa model mampu menjelaskan hampir seluruh variansi data target.
- **ExtraTrees** dan **RandomForest** juga tampil cukup kuat, meskipun sedikit di bawah dua model boosting tersebut.
- Model seperti **AdaBoost**, **Linear Regression**, dan **SVR** memiliki performa paling rendah, menunjukkan bahwa mereka kurang cocok untuk data yang kompleks seperti dalam kasus turbin angin ini.
- Nilai **MAPE** yang sangat besar secara absolut menunjukkan bahwa skala target variabel cukup besar atau terdapat nilai mendekati nol yang memperbesar error relatif (perlu pengecekan lebih lanjut jika MAPE akan digunakan untuk keputusan akhir).

### ✅ Kesimpulan:

Berdasarkan evaluasi, **CatBoost Regressor** dipilih sebagai model terbaik karena menghasilkan prediksi paling akurat (R² tertinggi dan RMSE/MAE terendah). Selain itu, model ini juga mampu menangani kompleksitas dan non-linearitas data dengan sangat baik tanpa perlu banyak preprocessing.

## 🛠️ Fine Tuning Model (CatBoost Regressor)

Berdasarkan hasil evaluasi awal, model **CatBoost Regressor** menunjukkan performa terbaik dalam memprediksi daya listrik yang dihasilkan oleh turbin angin. Untuk lebih mengoptimalkan kinerjanya, dilakukan proses fine-tuning terhadap hyperparameter model.

### 🎯 Tujuan Fine Tuning

Tujuan utama fine tuning adalah untuk mencari kombinasi parameter terbaik yang meminimalkan error prediksi. Proses ini dilakukan menggunakan **RandomizedSearchCV** dari scikit-learn dengan metrik evaluasi berupa **Root Mean Squared Error (RMSE)**.

### 🧪 Hyperparameter yang Diuji

- `learning_rate`: Tingkat pembelajaran untuk proses boosting.
- `iterations`: Jumlah pohon yang dibangun (estimators).
- `depth`: Kedalaman maksimum dari setiap pohon.
- `subsample`: Proporsi data yang digunakan untuk setiap pohon.
- `colsample_bylevel`: Proporsi fitur yang digunakan di setiap level pohon.
- `l2_leaf_reg`: Regularisasi L2 untuk mengurangi overfitting.
- `min_child_samples`: Minimum jumlah sampel di simpul daun.

### 🔍 Proses dan Hasil Tuning

Tuning dilakukan dengan:

- `n_iter = 50` kombinasi parameter acak.
- `cv = 5` cross-validation untuk menjaga generalisasi.

Setelah tuning, model CatBoost dilatih kembali menggunakan kombinasi parameter terbaik (`best_params`). Evaluasi akhir dilakukan dengan membandingkan hasil prediksi pada data uji (X_test) terhadap nilai aktual (y_test).

### ✅ Hasil Evaluasi Akhir

| Metrik   | Nilai                 |
| -------- | --------------------- |
| R² Score | 0.9881578176597767    |
| RMSE     | 142.14934648611487    |
| MAE      | 59.72447828787168     |
| MAPE     | 5.304711773256123e+16 |

### 📊 Visualisasi Prediksi

Grafik berikut menunjukkan perbandingan antara:

- Daya aktual yang dihasilkan (`LV ActivePower`)
- Prediksi model (`Predictions`)
- Daya teoretis (`Theoretical_Power_Curve`)

Daya aktual dan prediksi menunjukkan kecocokan yang sangat baik terhadap kecepatan angin, mendekati nilai daya teoretis, menandakan bahwa model bekerja secara realistis dan akurat.

![Prediction Plot](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/prediction_plot.png?raw=true)

## Referensi

[1] Swart, R., Robinson, J., & Cohen, S. (2003). _Climate change and sustainable development: expanding the options_. Climate Policy, 3(1), S19–S40. https://doi.org/10.1016/j.clipol.2003.10.010
[2] Delgado, I., Fahim, M. (2021). _Wind Turbine Data Analysis and LSTM-Based Prediction in SCADA System_. Energies 2021, 14(1), 125. https://doi.org/10.3390/en14010125
