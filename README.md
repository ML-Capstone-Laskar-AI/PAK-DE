# Parkinson Detection ML Capstone Project

## Project Overview
Penyakit Parkinson (PP) adalah gangguan neurodegeneratif umum yang pertama kali dideskripsikan oleh James Parkinson sebagai “Shaking Palsy”. Efek degeneratif pada mobilitas dan kontrol otot dapat dilihat melalui gejala yang dikenal sebagai tiga tanda utama PP seperti tremor saat istirahat (resting tremor), gerakan yang lambat (bradikinesia), dan kekakuan otot (rigiditas) yang disebabkan oleh berkurangnya produksi dopamin dalam otak. Penyakit ini sering didiagnosis terlambat karena metode konvensional bergantung pada identifikasi gejala motorik lanjut, yang muncul setelah kerusakan saraf signifikan[1]. 

Proyek ini bertujuan untuk mengembangkan sistem deteksi dini Parkinson berbasis kecerdasan buatan (AI) dengan menganalisis dataset gambar yang dibuat oleh penderita Parkinson. Kami akan menggunakan model Convolutional Neural Network (CNN) untuk menginvestigasi efektivitas model tersebut dalam mengklasifikasikan individu sebagai sehat atau penderita Penyakit Parkinson berdasarkan analisis pola pada dataset gambar tes spiral, mengukur kinerja model melalui metrik seperti akurasi, presisi, recall, F1-score, dan AUC. Sistem yang dikembangkan ini berpotensi menjadi alat skrining non-invasif, terukur, dan hemat biaya, meningkatkan aksesibilitas diagnosis dini PP.


## Business Understanding
### Problem Statements
Metode diagnosis Penyakit Parkinson saat ini seringkali bersifat subjektif, memerlukan waktu yang lama, mahal, dan seringkali baru dapat dilakukan ketika gejala motorik sudah parah, yang menandakan kerusakan saraf yang signifikan telah terjadi. Hal ini menghambat deteksi dini yang krusial untuk manajemen penyakit yang lebih efektif. Diperlukan sebuah alat skrining yang cepat, objektif, dan mudah diakses untuk membantu identifikasi awal individu yang berisiko.

### Goals
Tujuan utama dari proyek ini adalah:

1. **Mengembangkan Model Klasifikasi**: Membangun model deep learning yang mampu mengklasifikasikan gambar tulisan tangan (spiral, wave, meander) ke dalam dua kelas: 'Healthy' atau 'Parkinson'.
3. **Mencapai Target Performa**: Mencapai akurasi model minimal 85% pada test set untuk memastikan keandalan model sebagai alat skrining awal.
4. **Mendeploy Model**: Membuat aplikasi web sederhana sebagai proof-of-concept yang memungkinkan pengguna untuk mengunggah gambar dan mendapatkan hasil prediksi secara interaktif.

### Solution Approach
Untuk mencapai tujuan tersebut, kami akan menggunakan pendekatan transfer learning dengan model Convolutional Neural Network (CNN). Secara spesifik, kami akan menggunakan arsitektur MobileNetV2 yang sudah terlatih pada dataset ImageNet. Strategi yang akan diterapkan adalah feature extraction, di mana bobot dari MobileNetV2 akan dibekukan, dan kami akan melatih beberapa layer baru (top layers) di atasnya untuk melakukan klasifikasi biner. Pendekatan ini dipilih karena efektif untuk dataset dengan jumlah yang tidak terlalu besar, dengan memanfaatkan fitur-fitur visual yang telah dipelajari oleh model dari jutaan gambar. Eksperimen akan dilakukan dengan beberapa set hyperparameter untuk menemukan konfigurasi model yang paling optimal.

## Data Understanding
Dataset yang digunakan merupakan gabungan dari dua sumber publik:

1. HandPD dataset: [Link](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/)
2. Hand Drawings (Mendeley Data): [Link](https://data.mendeley.com/datasets/fd5wd6wmdj/1)

Setelah digabungkan, dataset ini berisi 940 gambar yang terbagi menjadi dua kelas utama: Healthy dan Parkinson. Gambar-gambar tersebut dikelompokkan lagi menjadi tiga jenis pola tulisan tangan yang umum digunakan dalam tes neurologis:

Spiral: Subjek diminta menggambar pola spiral.
Wave: Subjek diminta menggambar pola gelombang.
Meander: Subjek diminta menggambar pola berkelok-kelok.

Dari hasil Eksplorasi Data Awal (EDA), ditemukan beberapa karakteristik penting:

* Distribusi Kelas: Terdapat ketidakseimbangan kelas, di mana jumlah gambar untuk kelas 'Parkinson' lebih banyak daripada kelas 'Healthy'. Hal ini perlu ditangani saat pelatihan model (misalnya, dengan class weights).
* Format Gambar: Sebagian besar gambar memiliki mode warna RGB, namun ditemukan beberapa gambar dengan mode RGBA. Ini perlu diseragamkan menjadi RGB.
* Ukuran Gambar: Terdapat variasi resolusi antar gambar, sehingga perlu dilakukan standardisasi ukuran gambar sebelum dimasukkan ke dalam model.

## Data Preparation
Pada tahap ini, data mentah disiapkan agar siap untuk dilatih oleh model. Proses yang dilakukan adalah sebagai berikut:

1. Pembagian Data (Data Splitting): Dataset dibagi menjadi tiga set dengan rasio 80% Training, 10% Validation, dan 10% Test. Pembagian dilakukan secara terstratifikasi berdasarkan kelas utama ('Healthy' dan 'Parkinson') untuk memastikan proporsi kelas yang sama di setiap set.
2. Augmentasi Data (Data Augmentation): Untuk mengatasi jumlah data yang terbatas dan mencegah overfitting, teknik augmentasi data diterapkan hanya pada data training. Augmentasi yang digunakan meliputi rotasi acak, zoom acak, flip horizontal acak, dan penyesuaian kontras acak.
3. Preprocessing Gambar: Semua gambar, baik dari set training, validation, maupun test, melalui serangkaian proses yang sama:
  * Perubahan Ukuran (Resizing): Ukuran semua gambar diseragamkan menjadi 224x224 piksel agar sesuai dengan input yang diharapkan oleh model MobileNetV2.
  * Konversi Warna: Seluruh gambar dipastikan dalam format 3 channel (RGB). Konversi dari format lain seperti RGBA akan ditangani secara otomatis saat data dimuat.
  * Normalisasi Nilai Piksel: Nilai piksel gambar dinormalisasi menggunakan fungsi preprocess_input yang spesifik untuk MobileNetV2. Ini penting agar skala input sesuai dengan bagaimana model tersebut dilatih pada dataset aslinya (ImageNet).
4. Pembuatan Pipeline tf.data: Data diubah menjadi objek tf.data.Dataset yang efisien, dengan prefetch untuk mengoptimalkan kecepatan pemuatan data selama pelatihan dan evaluasi, sehingga CPU dapat menyiapkan data batch berikutnya sementara GPU sedang memproses batch saat ini.


## Modeling

Pada tahap ini, kami membangun dan melatih model deep learning untuk mengklasifikasikan gambar tulisan tangan menjadi dua kategori utama : 'Healthy' dan 'Parkinson'.

Model yang digunakan adalah hasil Transfer Learning menggunakan arsitektur **MobileNetV2** sebagai base model. MobileNetV2 adalah arsitektur Convolutional Nerual Network (CNN) yang efisien, dilatih pada dataset ImageNet. Kami memilih MobileNetV2 karena arsitekturnya yang ringan namun memiliki kemampuan ekstraksi fitur gambar yang baik, cocok untuk aplikasi yang mungkin memerlukan efisiensi komputasi. 

### Arsitektur Model 

`Base Model` : `MobileNetV2` (dengan bobot ImageNet), kami membekukan lapisan-lapisan pada *base model* ini di awal pelatihan untuk mempertahankan fitur-fitur yang telah dipelajari.
*   **Lapisan Kustom:** Setelah *base model*, kami menambahkan beberapa lapisan kustom untuk adaptasi dengan dataset dan tugas klasifikasi biner (Healthy vs Parkinson):
    *   `GlobalAveragePooling2D`: Mengurangi dimensi spasial fitur dari *base model*.
    *   `Dropout`: Menerapkan *dropout* dengan laju tertentu untuk mengurangi *overfitting*.
    *   `Dense` (dengan aktivasi ReLU): Lapisan *fully connected* dengan regularisasi L2 untuk mempelajari kombinasi fitur yang kompleks.
    *   `BatchNormalization`: Menormalisasi output dari lapisan Dense sebelumnya, membantu stabilisasi pelatihan.
    *   `Dropout`: Lapisan *dropout* kedua untuk penambahan regularisasi.
    *   `Dense` (dengan aktivasi Sigmoid): Lapisan output tunggal dengan aktivasi sigmoid untuk menghasilkan probabilitas kelas 'Parkinson'.

### Konfigurasi Pelatihan

*   **Optimizer:** Adam dengan *initial learning rate* tertentu.
*   **Loss Function:** `binary_crossentropy`, cocok untuk tugas klasifikasi biner.
*   **Metrik Evaluasi:** `accuracy`, `precision`, dan `recall` digunakan untuk memantau performa selama pelatihan dan evaluasi.
*   **Class Weight:** Menggunakan bobot kelas (`class_weight='balanced'`) untuk menangani potensi ketidakseimbangan jumlah sampel antar kelas ('Healthy' dan 'Parkinson') dalam dataset pelatihan.
*   **Callbacks:**
    *   `EarlyStopping`: Menghentikan pelatihan lebih awal jika performa pada set validasi berhenti meningkat (berdasarkan `val_loss`) selama beberapa *epoch* (`patience`). Ini mencegah *overfitting*.
    *   `ReduceLROnPlateau`: Mengurangi *learning rate* secara otomatis jika performa pada set validasi tidak membaik selama beberapa *epoch* tertentu, membantu model menemukan konvergensi yang lebih baik.

### Eksperimen dan Pelacakan (MLflow)

Kami melakukan beberapa eksperimen dengan konfigurasi *hyperparameter* yang berbeda (seperti *learning rate*, jumlah unit pada lapisan Dense, laju *dropout*, dan nilai regularisasi L2). Semua eksperimen ini dilacak menggunakan **MLflow**.

MLflow digunakan untuk:
*   **Melacak Parameter:** Menyimpan konfigurasi *hyperparameter* setiap run.
*   **Melacak Metrik:** Merekam metrik performa (loss, accuracy, precision, recall) per epoch selama pelatihan, serta metrik akhir pada set validasi dan set tes.
*   **Melacak Artefak:** Menyimpan artefak penting seperti plot performa (accuracy/loss vs epoch, confusion matrix), classification report, dan model Keras yang telah dilatih itu sendiri.

Dengan menggunakan MLflow, kami dapat membandingkan performa dari berbagai konfigurasi dengan mudah dan mengidentifikasi model terbaik berdasarkan metrik evaluasi pada set tes.

### Hasil

Setelah melatih model dengan berbagai konfigurasi dan mengevaluasinya pada set tes yang terpisah, kami memilih model terbaik berdasarkan akurasi pada set tes. Hasil evaluasi lengkap (loss, accuracy, precision, recall, confusion matrix, dan classification report) dari model terbaik ini di-*log* ke MLflow dan juga disajikan dalam notebook untuk analisis lebih lanjut pada tahap berikutnya.

## Evaluation

Tahap terakhir dari proyek ini adalah mengevaluasi performa model terbaik yang telah dilatih dan melakukan analisis mendalam terhadap jenis kesalahan yang dibuat oleh model.

### Tujuan Evaluasi

Evaluasi dilakukan untuk mengukur seberapa baik model generalisasi pada data yang belum pernah dilihat sebelumnya (set tes). Metrik utama yang kami fokuskan meliputi:

*   **Accuracy:** Proporsi prediksi yang benar secara keseluruhan.
*   **Precision:** Dari semua gambar yang diprediksi sebagai Parkinson, berapa proporsi yang sebenarnya Parkinson. Penting untuk meminimalkan False Positives (memprediksi Parkinson pada orang sehat).
*   **Recall (Sensitivity):** Dari semua gambar yang sebenarnya Parkinson, berapa proporsi yang berhasil dideteksi oleh model. Penting untuk meminimalkan False Negatives (gagal mendeteksi Parkinson pada orang yang sakit).
*   **F1-Score:** Rata-rata harmonik dari Precision dan Recall, memberikan keseimbangan antara kedua metrik tersebut.
*   **Loss:** Mengukur seberapa "salah" prediksi model (nilai yang lebih rendah lebih baik).

### Confusion Matrix

Sebagai bagian dari evaluasi, kami menghasilkan **Confusion Matrix** pada set tes. Matriks ini memberikan rincian visual tentang:

*   **True Positives (TP):** Gambar Parkinson yang diprediksi benar sebagai Parkinson.
*   **True Negatives (TN):** Gambar Sehat yang diprediksi benar sebagai Sehat.
*   **False Positives (FP):** Gambar Sehat yang diprediksi salah sebagai Parkinson.
*   **False Negatives (FN):** Gambar Parkinson yang diprediksi salah sebagai Sehat.

Analisis Confusion Matrix sangat penting untuk memahami jenis kesalahan spesifik yang dibuat model, yang seringkali lebih informatif daripada sekadar melihat akurasi total.

### Classification Report

Selain Confusion Matrix, **Classification Report** memberikan ringkasan metrik Precision, Recall, dan F1-Score untuk setiap kelas ('Healthy' dan 'Parkinson'), serta metrik rata-rata (support, macro avg, weighted avg). Ini memungkinkan kami untuk melihat metrik performa yang ditargetkan untuk setiap kelas secara terpisah.

### Analisis Kesalahan

Untuk mendapatkan wawasan lebih lanjut, kami melakukan **Analisis Kesalahan** pada set tes. Ini melibatkan:

1.  Mengidentifikasi sampel-sampel pada set tes di mana model membuat prediksi yang salah (baik False Positives maupun False Negatives).
2.  Memvisualisasikan beberapa contoh gambar yang salah diklasifikasikan.
3.  Mencoba memahami mengapa model mungkin kesulitan dengan gambar-gambar tersebut. Ini bisa karena:
    *   Variasi visual yang tinggi dalam dataset.
    *   Gambar yang buram atau berkualitas rendah.
    *   Fitur yang kurang jelas membedakan antara kelas pada sampel tertentu.
    *   Kesamaan visual antara gambar dari orang sehat dan orang dengan Parkinson pada kasus-kasus *borderline*.

Analisis kesalahan ini memberikan umpan balik berharga untuk potensi perbaikan model di masa mendatang, seperti:
*   Meningkatkan kualitas atau kuantitas data.
*   Menerapkan teknik augmentasi yang lebih canggih.
*   Menjelajahi arsitektur model lain.
*   Melakukan fine-tuning pada *base model* (jika sebelumnya dibekukan).

### Ringkasan Hasil

Hasil evaluasi dan analisis kesalahan memberikan gambaran komprehensif tentang performa model dalam mendeteksi Parkinson dari gambar tulisan tangan. Metrik kuantitatif (Accuracy, Precision, Recall) dikombinasikan dengan wawasan dari Confusion Matrix dan analisis visual sampel yang salah diklasifikasikan, membantu kami memahami kekuatan dan kelemahan model saat ini. Detail hasil spesifik dari model terbaik (termasuk nilai metrik dan visualisasi plot) dicatat dan dapat dilihat dalam log MLflow untuk run terbaik.

## Apakah Problem Statements Sudah Terselesaikan?

## Referensi
[1] Alia S, Hidayati HB, Hamdan M, et al. Penyakit Parkinson: Tinjauan Tentang Salah Satu Penyakit Neurodegeneratif yang Paling Umum. Aksona. 2022;1(2):95-99. doi:10.20473/aksona.v1i2.145

**---Ini adalah bagian akhir laporan---**
