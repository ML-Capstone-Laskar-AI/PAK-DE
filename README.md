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

## Evaluation

## Apakah Problem Statements Sudah Terselesaikan?

## Referensi
[1] Alia S, Hidayati HB, Hamdan M, et al. Penyakit Parkinson: Tinjauan Tentang Salah Satu Penyakit Neurodegeneratif yang Paling Umum. Aksona. 2022;1(2):95-99. doi:10.20473/aksona.v1i2.145

**---Ini adalah bagian akhir laporan---**
