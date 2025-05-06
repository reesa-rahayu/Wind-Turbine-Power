# Laporan Proyek Machine Learning - Rahayu Kartika Sari

## Domain proyek

![Wine Turbine](https://www.solarfeeds.com/mag/wp-content/uploads/2019/10/picspree-1256331.jpg)

Energi terbarukan, khususnya dari kincir angin, memainkan peran penting dalam transisi menuju sumber energi berkelanjutan [[1]](https://www.tandfonline.com/doi/abs/10.1016/j.clipol.2003.10.010). Dengan semakin meningkatnya kebutuhan energi dan kekhawatiran tentang perubahan iklim, penting untuk memanfaatkan teknologi yang ada untuk memprediksi dan mengoptimalkan hasil energi dari sumber ini.

Proyek ini bertujuan untuk menganalisis dan memprediksi energi yang dihasilkan oleh kincir angin menggunakan dataset yang berisi data SCADA (Supervisory Control and Data Acquisition) dari kincir angin di Turki pada tahun 2018.

### Mengapa dan Bagaimana Masalah Ini Dapat Dipecahkan

Dalam konteks dunia yang semakin menuntut penggunaan energi bersih, pemahaman yang lebih baik tentang variabel-variabel yang mempengaruhi output energi kincir angin dapat memberikan wawasan yang berharga bagi operator dan pengembang proyek energi terbarukan. Dengan menggunakan algoritma _Machine Learning_, proyek ini dapat membantu dalam menghasilkan model yang lebih akurat untuk memprediksi produksi energi berdasarkan faktor-faktor seperti kecepatan angin, suhu, dan kondisi lingkungan lainnya [[2]](https://www.mdpi.com/1996-1073/14/1/125).

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

- Membangun dan Membandingkan Beberapa Model Regresi untuk Prediksi Energi Kincir Angin
  Dilakukan percobaan dengan berbagai algoritma regresi, termasuk Gradient Boosting, SVR, Random Forest, Extra Trees, AdaBoost, Decision Tree, XGBoost, dan CatBoost untuk memprediksi daya aktif (LV ActivePower). Masing-masing model diuji pada data yang telah dibersihkan dan dinormalisasi, kemudian dievaluasi menggunakan metrik regresi seperti R² Score, RMSE, MAE, dan MAPE. Tujuannya adalah untuk mengidentifikasi model baseline dengan performa terbaik yang mampu menangkap hubungan non-linear dan kompleks antar variabel seperti kecepatan angin, suhu, dan arah angin.
- Melakukan exploratory data analysis (EDA) secara menyeluruh
  Analisis yang dilakukan termasuk analisis univariat (histogram dan boxplot), analisis multivariat (pairplot dan heatmap), serta analisis hubungan antar fitur terhadap target seperti kecepatan angin terhadap daya aktif. Proses ini membantu dalam mengidentifikasi pola musiman, siang/malam, serta outlier yang dapat memengaruhi performa model.
- Mengimplementasikan feature engineering
  Dilakukan ekstraksi informasi penting dari data waktu seperti jam, hari, bulan, musim, serta menambahkan fitur siang/malam menggunakan data astronomis wilayah Izmir (lokasi turbin), serta menambahkan fitur suhu dari data cuaca eksternal. Ini memberikan konteks yang lebih kaya terhadap perilaku turbin dalam berbagai kondisi lingkungan.
- Melakukan Hyperparameter Tuning untuk Meningkatkan Performa Model Terbaik
  Setelah model baseline dengan performa terbaik ditemukan, dilakukan RandomizedSearchCV untuk mencari kombinasi parameter optimal. Proses ini menggunakan metrik RMSE sebagai skor utama, untuk memastikan peningkatan kinerja model secara objektif dan terukur terhadap data uji. Hasil tuning menunjukkan peningkatan skor prediktif, mengindikasikan model mampu melakukan generalisasi yang lebih baik.

Dengan langkah-langkah ini, proyek bertujuan untuk mencapai efisiensi dan keakuratan tinggi dalam memprediksi energi yang dihasilkan oleh kincir angin, sehingga dapat memberikan informasi yang berguna untuk perencanaan dan manajemen sistem energi terbarukan.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [2018 SCADA Data of a Wind Turbine in Turkey](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset/data). Dataset ini berisi informasi operasional dari sebuah kincir angin selama periode tertentu, memperlihatkan berbagai variabel yang berkontribusi pada produksi energi. Data ini terukur pada interval waktu tertentu, mulai dari 1 Januari 2018 hingga 13 Desember 2018. Terdiri atas

### Variabel-variabel pada 2018 SCADA Data of a Wind Turbine in Turkey dataset adalah sebagai berikut:

- Date/Time: Stempel waktu yang mencatat tanggal dan waktu pengukuran dengan interval 10 menit.
- LV ActivePower (kW): Energi yang dihasilkan turbin yang menjadi variabel target
- Wind Speed (m/s): Kecepatan angin yang digunakan turbin untuk menghasilkan listrik
- Wind Direction (degrees): Arah angin yang diukur dalam derajat, membantu dalam memahami bagaimana orientasi kincir angin sehingga berpengaruh terhadap efisiensi.
- Theoretical_Power_Curve (KWh): Prediksi energi yang dihasilkan turbin secara teoritis.

### Exploratory Data Analysis (EDA) & Visualisasi

Analisis eksploratif dilakukan untuk memahami pola, distribusi, dan hubungan antar fitur dalam dataset.

1.  Visualisasi Sebaran dan kolerasi antara `LV ActivePower (kW)`,`Wind Speed (m/s)`, `Theoretical_Power_Curve (KWh)`, dan `Wind Direction (°)` menggunakan **Pairplot** dan **Histogram**.
    ![Pairplot](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/pairplot_observed_column.png?raw=true)

    ![Histogram](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/histogram_observes_column.png?raw=true)

2.  Distribusi Energi

    - **Siang vs Malam**: Malam hari menghasilkan energi lebih banyak secara akumulatif.
      ![Day/Night Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/accumulation_1.png?raw=true)
    - **Bulanan**: Produksi energi tertinggi terjadi di bulan **Maret** dan **Agustus**.
      ![Month Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/accumulation_2.png?raw=true)
    - **Musiman**: Energi paling banyak dihasilkan saat **musim gugur (Autumn)**.
      ![Season Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/accumulation_3.png?raw=true)

3.  Analisis Kecepatan Angin

    - Hubungan antara `Wind Speed (m/s)` dan:

      - **Theoretical Power** menunjukkan kurva daya ideal turbin.

        ![Wind Speed and Theoretical Power](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/power_1.png?raw=true)

      - **LV ActivePower** menunjukkan performa aktual.

        ![Wind Speed and Actual Power](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/power_2.png?raw=true)

    - Temuan penting:
      - Minimum kecepatan angin untuk menghasilkan daya ≈ **3.6 m/s**
      - Kecepatan maksimum sebelum daya stagnan ≈ **17.9 m/s**

4.  Analisis Arah Angin

    - Arah angin tersebar di semua sudut (0–360°), menunjukkan cakupan penuh oleh turbin.

      ![Wind Direction Accumulation](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/wind_direction.png?raw=true)

    - Arah angin paling optimal dalam menghasilkan energi adalah sekitar: **30–75°** dan **180–210°**

      ![Wind direction and Power](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/wind_direction_1.png?raw=true)

    - Arah angin ini juga berkorelasi dengan kecepatan angin yang tinggi.

      ![Wind direction and Wind Speed](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/wind_direction_2.png?raw=true)

5.  Korelasi Antar Fitur (Heatmap)
    ![Heatmap](https://github.com/reesa-rahayu/Wind-Turbine-Power/blob/main/images/heatmap.png?raw=true)

    - Korelasi kuat ditemukan antara `Wind Speed` dan `Theoretical_Power_Curve`, serta `Wind Speed` dengan `LV ActivePower`.
    - Visualisasi ini menegaskan bahwa kecepatan angin adalah fitur paling penting untuk prediksi daya yang dihasilkan.

## 🛠️ Data Preparation

Data Preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan machine learning bersih, konsisten, dan siap untuk dianalisis. Proses ini penting untuk meningkatkan kinerja model dengan mengatasi masalah potensial dalam kualitas data.

### Proses Data Preparation:

#### 1. 🧹 Data Cleaning

Tujuan: Membersihkan data dari kesalahan, nilai tak valid, dan ketidaksesuaian agar model tidak belajar dari informasi yang salah.

- Missing Data
  Nilai yang hilang dapat menyebabkan error dalam beberapa algoritma _machine learning_, menghasilkan bias jika tidak ditangani dengan tepat, atau menghilangkan informasi penting. Oleh karena itu, perlu dilakukan penanganan pada data-data ini. Berikut langkah yang dilakukan:

  - Dilakukan pemeriksaan jumlah nilai yang hilang pada setiap kolom menggunakan `.isna().sum()`.
  - Pada data asli sebelum feature engineering tidak terdapat nilai yang kosong.
  - Untuk kolom suhu (`Temperature (°C)`) yang didapatkan dari proses feature engineering, nilai yang kosong diatasi menggunakan metode **interpolasi**.
    Alasan: Interpolasi adalah metode yang masuk akal untuk mengisi nilai suhu yang hilang berdasarkan nilai-nilai suhu yang berdekatan dalam urutan waktu, sehingga dapat mempertahankan pola temporal data.

- Outlier Handling
  Outlier atau nilai-nilai yang berada jauh di luar distribusi normal dapat menyesatkan model dan menghasilkan prediksi yang tidak realistis, terutama pada algoritma yang sensitif terhadap nilai ekstrem seperti regresi linier, SVM, dan beberapa jenis neural networks. Oleh karena itu, deteksi dan penanganan outlier merupakan langkah penting dalam data preparation.

  - Daya listrik (`LV ActivePower (kW)` )
    Ditemukan nilai negatif pada kolom daya listrik (`LV ActivePower (kW)` ), padahal secara fisik turbin tidak mungkin menghasilkan energi negatif.

    - Penanganan:
      - Jumlah data anomali dihitung, kemudian semua nilai negatif diganti menjadi **0** menggunakan fungsi `.apply()`.
    - **Alasan**:

      - Mengganti nilai negatif dengan nol merupakan pendekatan yang konservatif dan masuk akal, karena mencerminkan situasi bahwa tidak ada energi yang dihasilkan pada saat itu.
      - Menghindari model belajar dari data yang tidak realistis dan menjaga validitas fisik data.

  - Kolom Wind Speed (m/s)

    - Dari hasil visualisasi menggunakan boxplot, ditemukan nilai kecepatan angin yang sangat tinggi yang tergolong sebagai outlier.
    - Secara teknis, kecepatan angin tinggi bisa saja terjadi secara alami dalam kondisi cuaca ekstrem seperti badai. Namun, perlu dikaji apakah nilai-nilai tersebut:
      - Masih berada dalam batas operasional turbin.
      - Cukup signifikan dalam jumlah untuk tetap dipertahankan.
    - Analisis Outlier dengan Metode IQR:

      - Metode Interquartile Range (IQR) digunakan untuk mendeteksi outlier secara statistik.
      - Nilai lower bound dan upper bound dihitung sebagai berikut:

        ```python
        Q1 = data['Wind Speed (m/s)'].quantile(0.25)
        Q3 = data['Wind Speed (m/s)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data['Wind Speed (m/s)'] < lower_bound) | (data['Wind Speed (m/s)'] > upper_bound)]
        outliers.shape[0]
        ```

      - Ditemukan bahwa hanya sekitar 423 data (kurang dari 1% dari total ~50.000 data) yang tergolong outlier menurut metode ini.

    - Penanganan:

      - Outlier tidak dihapus, melainkan diganti (capped) dengan nilai batas bawah (`lower_bound`) dan batas atas (`upper_bound`) menggunakan metode capping (winsorization):

      ```python
      data['Wind Speed (m/s)'] = np.where(data['Wind Speed (m/s)'] < lower_bound, lower_bound, data['Wind Speed (m/s)'])
      data['Wind Speed (m/s)'] = np.where(data['Wind Speed (m/s)'] > upper_bound, upper_bound, data['Wind Speed (m/s)'])

      ```

    - Alasan:
      - IQR adalah metode robust yang tidak terpengaruh oleh distribusi ekstrem, sehingga cocok untuk data yang tidak normal.
      - Metode capping dipilih untuk menjaga jumlah data, sambil membatasi pengaruh nilai ekstrem terhadap proses pelatihan model.
      - Karena jumlah outlier sangat kecil dan kemungkinan besar merupakan anomali atau noise sensor, mengganti nilai ekstrem ini dapat membantu model menjadi lebih stabil dan akurat.

#### 2. 🔧 **Feature Engineering**

Tahapan ini dilakukan untuk memperkaya dataset dengan fitur-fitur tambahan yang relevan, guna meningkatkan kualitas model prediksi energi angin. Fitur yang baik dapat membantu model untuk lebih baik memahami pola dalam data dan meningkatkan akurasi prediksi.

1.  **Ekstraksi Waktu**

    - Menambahkan kolom:
      - `Hour`: Jam ke berapa (0–23)
      - `Day`: Hari dalam bulan
      - `Week`: Minggu ke berapa dalam tahun
      - `Month`: Bulan ke berapa
    - Alasan: Untuk menangkap pola harian, mingguan, dan musiman yang berpotensi memengaruhi produksi energi angin.

2.  **Identifikasi Musim (Season)**

    - Menentukan musim berdasarkan nilai `Month`:
      - 1 = Winter (Des–Feb)
      - 2 = Spring (Mar–Mei)
      - 3 = Summer (Jun–Agu)
      - 4 = Autumn (Sep–Nov)
    - Alasan: Kecepatan angin dan produksi energi sangat dipengaruhi oleh perubahan musim, sehingga fitur ini memperkaya informasi kontekstual yang dapat dipelajari oleh model.

3.  **Penentuan Siang/Malam (Day/Night)**

    - Menggunakan library **Astral** untuk menghitung waktu matahari terbit dan terbenam di kota Izmir, Turki.
    - Menambahkan kolom `Day/Night`:
      - 0 = Siang
      - 1 = Malam
    - Alasan: Pola angin dapat berbeda signifikan antara siang dan malam hari. Fitur ini membantu model membedakan karakteristik produksi energi berdasarkan waktu hari.

4.  **Suhu Udara (Temperature)**
    - Alasan: Suhu berpengaruh terhadap densitas udara, yang secara langsung memengaruhi energi kinetik angin. Menambahkan fitur ini dapat meningkatkan akurasi prediksi daya turbin.
    - Data suhu diperoleh dari **Meteostat API** berdasarkan lokasi dan waktu.
    - Disesuaikan ke interval 10 menit agar sinkron dengan data utama.
    - Fitur `Temperature (°C)` ditambahkan dan nilai yang hilang diisi menggunakan interpolasi.

#### 3. **Exploratory Data Analysis (EDA) Lanjutan**

Setelah proses Feature Engineering selesai, dilakukan EDA lanjutan untuk menganalisis pengaruh fitur-fitur baru terhadap energi yang dihasilkan (LV ActivePower (kW)). Langkah ini bertujuan untuk menggali hubungan baru, menemukan pola tersembunyi, dan mendapatkan pemahaman yang lebih dalam mengenai perilaku sistem turbin angin berdasarkan konteks waktu, musim, dan kondisi lingkungan.

🎯 Tujuan EDA Lanjutan:

- Memvalidasi bahwa fitur hasil feature engineering memiliki nilai prediktif terhadap target (LV ActivePower).
- Membantu dalam pemilihan fitur untuk proses modeling.
- Menyediakan insight domain-spesifik tentang bagaimana kondisi waktu dan lingkungan memengaruhi output energi turbin.

#### 4. **Normalisasi**

Beberapa fitur dalam dataset, seperti Wind Speed dan Temperature, memiliki skala yang sangat berbeda. Misalnya, kecepatan angin dapat berkisar antara 0 hingga 25 m/s, sementara suhu dapat berkisar antara 0 hingga 40 °C. Model yang sensitif terhadap skala fitur (seperti regresi linier atau jaringan saraf) dapat dipengaruhi jika fitur dengan skala yang lebih besar mendominasi proses pelatihan. Oleh karena itu, penting untuk menormalkan fitur numerik.

**Langkah yang Diambil**:
Normalisasi dilakukan menggunakan **StandardScaler**, yaitu metode standardisasi yang mengubah distribusi setiap fitur menjadi memiliki nilai rata-rata 0 dan standar deviasi 1, menggunakan rumus:

![Standard Scaler Equation](https://journaldev.nyc3.cdn.digitaloceanspaces.com/2020/10/Standardization.png)

di mana $X$ adalah nilai asli, $\mu$ adalah rata-rata fitur, dan $\sigma$ adalah standar deviasi fitur.

Alasan Penggunaan:
Standardisasi digunakan karena metode ini bekerja baik dengan algoritma yang mengasumsikan distribusi data normal atau sensitif terhadap skala, seperti Regresi Linier, SVM, dan Neural Networks. Dengan fitur berada pada skala yang seragam, model dapat mempelajari pola dengan lebih stabil dan efisien.

#### 5. **Splitting Data (Train-Test Split)**

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

### ⚙️ Tahapan Pemodelan dan Parameter yang Digunakan

Untuk memprediksi daya listrik (LV ActivePower (kW)) yang dihasilkan turbin angin, proses pemodelan dilakukan dengan pendekatan eksperimen komparatif terhadap 10 algoritma regresi populer. Tujuannya adalah mengidentifikasi model dengan performa terbaik berdasarkan data fitur yang tersedia seperti kecepatan angin, suhu, dan parameter operasional lainnya.

1. Inisialisasi Model
   Dilakukan inisiasi baseline model dengan parameter awal sama yaitu `random_state=42`. Sepuluh model regresi dari berbagai kategori (linear, pohon keputusan, ensemble, dan boosting) digunakan:

```python
models = [
  GradientBoostingRegressor(random_state=42),
  SVR(),
  RandomForestRegressor(random_state=42),
  LinearRegression(),
  ExtraTreesRegressor(random_state=42),
  AdaBoostRegressor(random_state=42),
  DecisionTreeRegressor(random_state=42),
  XGBRegressor(random_state=42),
  XGBRFRegressor(random_state=42),
  CatBoostRegressor(random_state=42)
]
```

#### 📌 Model yang Digunakan

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

2. Training dan Evaluasi
   Setiap model dilatih menggunakan data pelatihan (X_train, y_train) dan diuji terhadap data pengujian (X_test, y_test). Hasil evaluasi disimpan dalam daftar.

   🧪 **Hasil Training Model**

   Setelah proses pelatihan dan evaluasi dilakukan terhadap berbagai algoritma regresi, berikut adalah perbandingan performa masing-masing model berdasarkan empat metrik evaluasi:

   | Model                     | R² Score   | RMSE       | MAE          | MAPE         |
   | ------------------------- | ---------- | ---------- | ------------ | ------------ |
   | CatBoostRegressor         | 0.984136   | 164.526242 | 72.580935    | 7.378838e+16 |
   | XGBRegressor 0.983315     | 168.731432 | 73.877128  | 7.464262e+16 |
   | ExtraTreesRegressor       | 0.978385   | 192.048116 | 69.812715    | 6.989515e+16 |
   | RandomForestRegressor     | 0.975389   | 204.925309 | 73.270102    | 7.666696e+16 |
   | DecisionTreeRegressor     | 0.957625   | 268.897146 | 88.132662    | 5.869001e+16 |
   | GradientBoostingRegressor | 0.951877   | 286.554685 | 127.704954   | 1.544708e+17 |
   | XGBRFRegressor            | 0.945945   | 303.701156 | 123.869380   | 1.482155e+17 |
   | LinearRegression          | 0.905880   | 400.747716 | 207.871353   | 3.014076e+17 |
   | SVR                       | 0.895137   | 423.000772 | 165.704126   | 2.368704e+17 |
   | AdaBoostRegressor         | 0.885727   | 441.571286 | 273.100957   | 1.999998e+17 |

   🔍 **Interpretasi Hasil**:

   - **CatBoost Regressor** dan **XGBoost Regressor** menunjukkan performa terbaik dengan nilai R² mendekati 1, menunjukkan bahwa model mampu menjelaskan hampir seluruh variansi data target.
   - **ExtraTrees** dan **RandomForest** juga tampil cukup kuat, meskipun sedikit di bawah dua model boosting tersebut.
   - Model seperti **AdaBoost**, **Linear Regression**, dan **SVR** memiliki performa paling rendah, menunjukkan bahwa mereka kurang cocok untuk data yang kompleks seperti dalam kasus turbin angin ini.
   - Nilai **MAPE** yang sangat besar secara absolut menunjukkan bahwa skala target variabel cukup besar atau terdapat nilai mendekati nol yang memperbesar error relatif (perlu pengecekan lebih lanjut jika MAPE akan digunakan untuk keputusan akhir).

3. Pemilihan Model Terbaik ✅

   Berdasarkan evaluasi, **CatBoost Regressor** dipilih sebagai model terbaik karena menghasilkan prediksi paling akurat (R² tertinggi dan RMSE/MAE terendah). Selain itu, model ini juga mampu menangani kompleksitas dan non-linearitas data dengan sangat baik tanpa perlu banyak preprocessing.

4. 🛠️ Fine Tuning Model

   Berdasarkan hasil evaluasi awal, model **CatBoost Regressor** menunjukkan performa terbaik dalam memprediksi daya listrik yang dihasilkan oleh turbin angin. Untuk lebih mengoptimalkan kinerjanya, dilakukan proses fine-tuning terhadap hyperparameter model.

   Tujuan utama fine tuning adalah untuk mencari kombinasi parameter terbaik yang meminimalkan error prediksi. Proses ini dilakukan menggunakan **RandomizedSearchCV** dari scikit-learn dengan metrik evaluasi berupa **Root Mean Squared Error (RMSE)**.

   🧪 **Hyperparameter yang Diuji**

   - `learning_rate`: Tingkat pembelajaran untuk proses boosting.
   - `iterations`: Jumlah pohon yang dibangun (estimators).
   - `depth`: Kedalaman maksimum dari setiap pohon.
   - `subsample`: Proporsi data yang digunakan untuk setiap pohon.
   - `colsample_bylevel`: Proporsi fitur yang digunakan di setiap level pohon.
   - `l2_leaf_reg`: Regularisasi L2 untuk mengurangi overfitting.
   - `min_child_samples`: Minimum jumlah sampel di simpul daun.

   🔍 **Proses dan Hasil Tuning**

   Tuning dilakukan dengan:

   - `n_iter = 50` kombinasi parameter acak.
   - `cv = 5` cross-validation untuk menjaga generalisasi.

   Setelah tuning, model CatBoost dilatih kembali menggunakan kombinasi parameter terbaik (`best_params`). Evaluasi akhir dilakukan dengan membandingkan hasil prediksi pada data uji (X_test) terhadap nilai aktual (y_test).

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

### ✅ Hasil Evaluasi Model Akhir

| Metrik   | Nilai                  |
| -------- | ---------------------- |
| R² Score | 0.9875110681229394     |
| RMSE     | 145.97942337297064     |
| MAE      | 61.40934968194007      |
| MAPE     | 5.6951631630036136e+16 |

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
