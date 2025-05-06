# Laporan Proyek Machine Learning - Rahayu Kartika Sari

## Domain proyek

![Wine Turbine](https://www.solarfeeds.com/mag/wp-content/uploads/2019/10/picspree-1256331.jpg)

Energi terbarukan, khususnya dari kincir angin, memainkan peran penting dalam transisi menuju sumber energi berkelanjutan [1][2]. Dengan semakin meningkatnya kebutuhan energi dan kekhawatiran tentang perubahan iklim, penting untuk memanfaatkan teknologi yang ada untuk memprediksi dan mengoptimalkan hasil energi dari sumber ini.

Proyek ini bertujuan untuk menganalisis dan memprediksi energi yang dihasilkan oleh kincir angin menggunakan dataset yang berisi data SCADA (Supervisory Control and Data Acquisition) dari kincir angin di Turki pada tahun 2018.

### Mengapa dan Bagaimana Masalah Ini Dapat Dipecahkan

Dalam konteks dunia yang semakin menuntut penggunaan energi bersih, pemahaman yang lebih baik tentang variabel-variabel yang mempengaruhi output energi kincir angin dapat memberikan wawasan yang berharga bagi operator dan pengembang proyek energi terbarukan. Dengan menggunakan algoritma _Machine Learning_, proyek ini dapat membantu dalam menghasilkan model yang lebih akurat untuk memprediksi produksi energi berdasarkan faktor-faktor seperti kecepatan angin, suhu, dan kondisi lingkungan lainnya.

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
   1. Missing Data
      - Dilakukan pemeriksaan jumlah nilai yang hilang pada setiap kolom menggunakan `.isna().sum()`.
      - Untuk kolom suhu (`Temperature (°C)`), nilai yang hilang sebelumnya telah diatasi menggunakan metode **interpolasi** saat proses feature engineering.

   2. Outlier Handling
      - Pada kolom `LV ActivePower (kW)` ditemukan nilai **negatif**, yang secara fisik tidak mungkin terjadi karena turbin tidak dapat menghasilkan daya negatif.
      - Jumlah data anomali dihitung, kemudian semua nilai negatif diganti menjadi **0** menggunakan fungsi `.apply()`.
      - Langkah ini penting untuk menjaga validitas data dan mencegah model belajar dari informasi yang salah.

#### 🔧 **Feature Engineering**
   Tahapan ini dilakukan untuk memperkaya dataset dengan fitur-fitur tambahan yang relevan, guna meningkatkan kualitas model prediksi energi angin.

   1. **Ekstraksi Waktu**
      - Menambahkan kolom:
      - `Hour`: Jam ke berapa (0–23)
      - `Day`: Hari dalam bulan
      - `Week`: Minggu ke berapa dalam tahun
      - `Month`: Bulan ke berapa
      - Bertujuan menangkap pola harian dan musiman dalam data.

   2. **Identifikasi Musim (Season)**
      - Menentukan musim berdasarkan nilai `Month`:
      - 1 = Winter (Des–Feb)
      - 2 = Spring (Mar–Mei)
      - 3 = Summer (Jun–Agu)
      - 4 = Autumn (Sep–Nov)
      - Fitur ini membantu memahami dampak perubahan musim terhadap kecepatan angin.

   3. **Penentuan Siang/Malam (Day/Night)**
      - Menggunakan library **Astral** untuk menghitung waktu matahari terbit dan terbenam di kota Izmir, Turki.
      - Menambahkan kolom `Day/Night`:
      - 0 = Siang
      - 1 = Malam
      - Bertujuan membedakan perilaku angin antara siang dan malam hari.

   4. **Suhu Udara (Temperature)**
      - Data suhu diperoleh dari **Meteostat API** berdasarkan lokasi dan waktu.
      - Disesuaikan ke interval 10 menit agar sinkron dengan data utama.
      - Fitur `Temperature (°C)` ditambahkan dan nilai yang hilang diisi menggunakan interpolasi.
      - Suhu digunakan karena berpengaruh terhadap densitas udara yang berperan dalam rumus energi kinetik angin.

#### 📊 Data Analysis & Visualisasi

Analisis eksploratif dilakukan untuk memahami pola, distribusi, dan hubungan antar fitur dalam dataset.

   1. Analisis Distribusi dan Korelasi
      - Visualisasi awal menggunakan **Pairplot** dan **Histogram** menunjukkan sebaran dan korelasi antara:
         - `LV ActivePower (kW)`
         - `Wind Speed (m/s)`
         - `Theoretical_Power_Curve (KWh)`
         - `Wind Direction (°)`
         - `Temperature (°C)`
      - **Boxplot** digunakan untuk mengidentifikasi outlier, terutama pada kolom `Wind Speed (m/s)`.

   2. Pola Data terhadap Waktu
      - Plot garis menunjukkan pola fluktuasi fitur terhadap waktu.
      - Memberikan wawasan tentang musim, harian, dan perbedaan waktu (siang vs malam) dalam produksi energi.

   3. Distribusi Energi
      - **Siang vs Malam**: Malam hari menghasilkan energi lebih banyak secara akumulatif.
      - **Bulanan**: Produksi energi tertinggi terjadi di bulan **Maret** dan **Agustus**.
      - **Musiman**: Energi paling banyak dihasilkan saat **musim gugur (Autumn)**.

   4. Analisis Kecepatan Angin
      - Hubungan antara `Wind Speed (m/s)` dan:
      - **Theoretical Power** menunjukkan kurva daya ideal turbin.
      - **LV ActivePower** menunjukkan performa aktual.
      - Ditemukan:
      - Kecepatan minimum untuk menghasilkan daya ≈ **3.6 m/s**
      - Kecepatan minimum untuk mencapai daya maksimum ≈ **13.8 m/s**

   5. Analisis Arah Angin
      - Arah angin tersebar di semua sudut (0–360°), menunjukkan cakupan penuh oleh turbin.
      - Arah angin paling optimal dalam menghasilkan energi adalah sekitar:
      - **30–75°** dan **180–210°**
      - Arah angin ini juga berkorelasi dengan kecepatan angin yang tinggi.

   6. Korelasi Antar Fitur (Heatmap)
      - Korelasi kuat ditemukan antara `Wind Speed` dan `Theoretical_Power_Curve`, serta `Wind Speed` dengan `LV ActivePower`.
      - Visualisasi ini menegaskan bahwa kecepatan angin adalah fitur paling penting untuk prediksi daya yang dihasilkan.


## Data Preparation

Data Preparation dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan machine learning bersih, konsisten, dan siap untuk dianalisis. Proses ini penting untuk meningkatkan kinerja model dengan mengatasi masalah potensial dalam kualitas data.

### Proses Data Preparation:

1. **Normalisasi**
   Masalah Skala Fitur: Beberapa fitur dalam dataset, seperti Wind Speed dan Temperature, memiliki skala yang sangat berbeda. Misalnya, kecepatan angin dapat berkisar antara 0 hingga 25 m/s, sementara suhu dapat berkisar antara 0 hingga 40 °C. Model yang sensitif terhadap skala fitur (seperti regresi linier atau jaringan saraf) dapat dipengaruhi jika fitur dengan skala yang lebih besar mendominasi proses pelatihan. Oleh karena itu, penting untuk menormalkan fitur numerik.

   **Langkah yang Diambil**:
   Normalisasi dilakukan dengan metode **Min-Max Scaling** untuk fitur-fitur numerik dalam dataset. Proses ini mengubah nilai setiap fitur ke dalam rentang [0, 1] dengan menggunakan rumus:

   ```math
   X' = \frac{X - X_{min}}{X_{max} - X_{min}}
   ```

   di mana $X$ adalah nilai asli, $X_{min}$ adalah nilai minimum dalam fitur, dan $X_{max}$ adalah nilai maksimum dalam fitur.

   Alasan Penggunaan:
   Normalisasi diperlukan untuk mengatasi masalah skala variabel yang berbeda, sehingga algoritma machine learning tidak bias terhadap variabel dengan nilai yang lebih besar. Dalam konteks ini, fitur seperti Wind Speed (kecepatan angin) dan Temperature (suhu) dapat memiliki rentang nilai yang jauh berbeda, dan algoritma yang sensitif terhadap skala fitur (seperti regresi atau jaringan saraf) mungkin memberikan bobot lebih pada fitur dengan nilai lebih tinggi. Dengan normalisasi, semua fitur berada dalam skala yang sama, membantu model beroperasi lebih efisien.

2. **Splitting Data (Train-Test Split)**

   Pemisahan data untuk pelatihan dan pengujian penting dilakukan untuk menghindari overfitting, di mana model hanya mengingat data pelatihan. Dataset dibagi menjadi 80% data pelatihan dan 20% data pengujian. Pemisahan ini memungkinkan penilaian kinerja model yang lebih akurat ketika dihadapkan pada data yang belum pernah dilihat sebelumnya. Pembagian ini bertujuan untuk mendapatkan gambaran yang jelas mengenai performa model dalam konteks dunia nyata.

## Modeling

Model machine learning yang digunakan dalam proyek ini adalah model prediksi, yang fokus pada memprediksi energi yang dihasilkan dari kincir angin berdasarkan fitur-fitur yang tersedia. Algoritma yang akan dieksplorasi meliputi:

1. **ARIMA** (Autoregressive Integrated Moving Average)
   ARIMA adalah model statistika yang digunakan untuk analisis dan prediksi deret waktu. Model ini menggabungkan komponen autoregresif (AR), perbedaan (I) untuk membuat data stasioner, dan rata-rata bergerak (MA). ARIMA efektif dalam mengidentifikasi pola dalam data historis dan membuat prediksi berdasarkan pola tersebut.

   Kelebihan:

   - Mampu menangkap pola temporal dalam data deret waktu.
   - Sangat baik dalam memprediksi data yang memiliki tren atau musiman.
   - Relatif mudah untuk diinterpretasikan dan diterapkan pada data yang stasioner.

   Kekurangan:

   - Membutuhkan data yang stasioner; jika data tidak stasioner, transformasi perlu dilakukan.
   - Tidak selalu cocok untuk menangkap hubungan kompleks di antara banyak fitur.

2. **RNN**(Recurrent Neural Network)
   RNN adalah jenis jaringan saraf yang dirancang untuk memproses urutan data dengan memanfaatkan memori dari langkah sebelumnya. Struktur RNN memudahkan model untuk mempertahankan informasi dari input sebelumnya dan menggunakannya untuk memprediksi output saat ini.

   Kelebihan:

   - Mampu mengolah data urutan atau sekuensial, sangat baik dalam menangkap dependensi temporal.
   - Dapat belajar dari data yang lebih panjang tanpa perlu fitur rekayasa yang rumit.

   Kekurangan:

   - Rentan terhadap masalah vanishing gradient, membuat pelatihan untuk urutan yang panjang menjadi sulit.
   - Memerlukan waktu pelatihan yang lebih lama dibandingkan model tradisional.

3. **LSTM** (Long Short-Term Memory)
   LSTM adalah variasi dari RNN yang dirancang untuk mengatasi masalah _vanishing gradient_. LSTM menggunakan memori sel yang memungkinkan model menyimpan informasi untuk periode waktu yang lebih lama dan menjadikannya lebih bertahan terhadap perubahan dalam pola input.

   Kelebihan:

   - Khusus dirancang untuk mengatasi masalah pada RNN, mampu mengingat informasi lebih lama.
   - Mampu memodelkan hubungan kompleks dalam data urutan, sehingga ideal untuk data time series.

   Kekurangan:

   - Lebih kompleks dan membutuhkan lebih banyak compute power daripada RNN konvensional.
   - Memerlukan banyak data untuk pelatihan yang efektif.

4. **Bi-LSTM** (Bidirectional Long Short-Term Memory)
   Bi-LSTM merupakan pengembangan dari LSTM yang memungkinkan model untuk membaca data dari kedua arah (maju dan mundur). Ini memberikan konteks yang lebih baik dengan memanfaatkan informasi dari masa lalu dan masa depan.

   Kelebihan:

   - Mampu menangkap lebih banyak informasi konteks dalam data urutan, meningkatkan akurasi prediksi.
   - Lebih baik dalam menangani urutan yang memiliki informasi penting dari kedua arah waktu.

   Kekurangan:

   - Memerlukan lebih banyak sumber daya komputasi dan waktu pelatihan dibandingkan dengan model LSTM standar.
   - Strukturnya yang lebih kompleks dapat membuat interpretasi model menjadi sulit.

5. **GRU** (Gated Recurrent Unit)
   GRU adalah jenis RNN yang lebih sederhana dibandingkan LSTM tetapi tetap efektif dalam menangkap informasi temporal. GRU menggabungkan beberapa fungsi dalam satu gerbang, sehingga memiliki arsitektur yang lebih efisien.

   Kelebihan:

   - Memiliki lebih sedikit parameter daripada LSTM, sehingga lebih cepat dalam pelatihan dan memakan lebih sedikit memori.
   - Efektif dalam menangkap pola temporal yang ada tanpa membutuhkan banyak lapisan.

   Kekurangan:

   - Meskipun lebih sederhana, performa GRU dapat sedikit lebih rendah dalam kasus-kasus tertentu dibandingkan dengan LSTM.
   - Belum sepopuler LSTM, sehingga terdapat kurangnya pengetahuan dan teknik yang dioptimalkan untuk GRU.

## Evaluation

Setelah melakukan pelatihan model menggunakan dataset yang sudah dipersiapkan, evaluasi dilakukan untuk menilai kinerja model-model yang telah dibangun. Evaluasi ini menggunakan beberapa metrik yang penting untuk masalah regresi kontinu karena fokus kita adalah memprediksi output daya yang dihasilkan dari kincir angin.

### Metrik Evaluasi yang Digunakan:

1. **Mean Absolute Error (MAE)**

   MAE menghitung rata-rata dari kesalahan absolut antara nilai prediksi dan nilai aktual. Ini memberikan gambaran seberapa banyak kesalahan dalam prediksi dalam satuan yang sama dengan data.

   Rumus:

   $$
   \text{MAE} = \frac{1}{n} \sum\_{i=1}^{n} |y_i - \hat{y_i}|
   $$

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

   $$
   \text{MAPE} = \frac{100\%}{n} \sum\_{i=1}^{n} \left|\frac{y_i - \hat{y_i}}{y_i}\right|
   $$

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

   $$
   \text{RMSE} = \sqrt{\frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y_i})^2}
   $$

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

   $$
   R^2 = 1 - \frac{\sum*{i=1}^{n} (y_i - \hat{y_i})^2}{\sum*{i=1}^{n} (y_i - \bar{y})^2}
   $$

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

### Hasil Evaluasi:

1. Logistic Regression:
   - Akurasi: 0.90 (Test)
   - Precision: 0.89 (Test)
   - Recall: 0.85 (Test)
   - F1-score: 0.80 (Test)
   - ROC-AUC: 0.92 (Test)
2. Decision Tree:

   - Akurasi: 1.0 (Test)
   - Precision: 1.0 (Test)
   - Recall: 1.0 (Test)
   - F1-score: 1.0 (Test)
   - ROC-AUC: 1.0 (Test)

3. Random Forest:

   - Akurasi: 1.0 (Test)
   - Precision: 1.0 (Test)
   - Recall: 1.0 (Test)
   - F1-score: 1.0 (Test)
   - ROC-AUC: 1.0 (Test)

4. XGBoost:
   - Akurasi: 1.0 (Test)
   - Precision: 1.0 (Test)
   - Recall: 1.0 (Test)
   - F1-score: 1.0 (Test)
   - ROC-AUC: 1.0 (Test)

Berdasarkan hasil evaluasi, model **XGBoost** memberikan hasil terbaik dengan 100% di semua metrik (Akurasi, Precision, Recall, F1-Score, dan ROC AUC). Oleh karena itu, XGBoost dipilih sebagai model terbaik untuk digunakan dalam mengidentifikasi PCOS. Keunggulannya dalam menangani data yang kompleks dan tidak seimbang menjadikannya pilihan yang tepat, terutama di bidang medis yang memerlukan akurasi tinggi dan kemampuan untuk menggeneralisasi.

Namun, perlu diwaspadai potensi overfitting pada model ini.
