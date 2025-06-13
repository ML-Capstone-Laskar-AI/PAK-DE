# Website-PakDe

-----

# 👋 **PakDe - Parkinson Detection: Deteksi Parkinson Lebih Awal, Hidup Lebih Berkualitas\!** 🧠✨


**Capek nebak-nebak gejala Parkinson? 🤔 PakDe hadir sebagai sahabat cerdasmu\!** 🤖❤️

PakDe (Parkinson Detection) adalah platform deteksi penyakit Parkinson berbasis **kecerdasan buatan** yang dirancang untuk membantu mendeteksi gejala awal penyakit Parkinson melalui analisis gambar tulisan tangan (spiral 🌀, meander 〰️, atau gelombang 🌊). **Deteksi dini itu penting banget lho, biar penanganan bisa lebih efektif dan kualitas hidup tetap terjaga\!** 💪

## 🚀 **Kenapa PakDe Ini Keren?**

Penyakit Parkinson itu kayak "ninja" 🥷, geraknya pelan tapi pasti ganggu sistem saraf dan gerakan tubuh. Sayangnya, deteksinya seringkali telat ⏰, nunggu gejalanya udah parah. Padahal, kalau ketahuan lebih awal, penanganannya bisa lebih oke dan hidup jadi lebih nyaman\! 😊

Nah, **PakDe** hadir dengan kekuatan **AI** dari **MobileNetV2** buat menganalisis gambar tulisan tanganmu. Jadi, kamu tinggal upload gambar, dan **PakDe** langsung kasih tau hasilnya dalam sekejap\! ⚡️ Webnya juga super gampang dipake lagi\! 😉

## ✨ **Fitur-Fitur Andalan PakDe:**

  - **🔍 Deteksi Parkinson Cepat:** Upload gambar spiral, meander, atau gelombang tulisan tanganmu ✍️, dan biarkan AI pintar PakDe menganalisis kemungkinan adanya Parkinson. Hasilnya langsung nongol\! 🤩
  - **📚 Gudang Informasi Parkinson:** Pengen tau lebih banyak soal Parkinson? Tenang\! Ada halaman lengkap soal gejala 🤕, tahapan \<0xF0\>\<0x9F\>\<0xA7\>\<0xAE\>, pengobatan 💊, dan cara pencegahannya. Biar kamu makin *aware*\! 👍
  - **📰 Jurnal Pintar:** Buat kamu yang pengen lebih dalam, ada juga nih kumpulan jurnal dan publikasi ilmiah paling *update* soal Parkinson dan cara deteksinya. Ilmu itu penting\! 🤓
  - **🧑‍💻 Tim Kreatif di Balik PakDe:** Kenalan yuk sama tim kece yang udah bikin PakDe jadi kenyataan\! Ada info lengkap tentang kami di halaman tim. 👋

## 🛠️ **Teknologi Canggih Ala PakDe:**

  - **Backend:** Python dengan framework **Flask** 🐍
  - **Frontend:** HTML, CSS, JavaScript, **Bootstrap 5** ✨
  - **Otak Pintar:** **TensorFlow** dengan arsitektur **MobileNetV2** 🧠
  - **Biar Makin Ciamik:** **AOS (Animate On Scroll) Library** untuk animasinya 💫
  - **Ikon Kece:** **Font Awesome** buat ikon-ikon yang menarik

## ⚙️ **Yuk, Cobain PakDe\! (Cara Instalasi):**

### 🎯 **Yang Kamu Butuhkan:**

  - Python versi 3.8 ke atas 🐍
  - pip (si tukang install paket Python 📦)

### 👇 **Langkah-Langkahnya:**

1.  Ambil dulu "rumah" PakDe (clone atau download repository ini) 🏠.

2.  Buka terminal atau command prompt (si jendela ajaib 🧙‍♂️).

3.  Masuk ke dalam folder PakDe:
    \`\`\`bash
    cd path/to/pakde
    \`\`\`

4.  Aktifin dulu "alam virtual" biar nggak bentrok sama yang lain:
    \`\`\`bash

    # Kalau pake .venv

    .venv\\Scripts\\activate

    # Atau kalau pake venv

    venv\\Scripts\\activate
    \`\`\`

5.  Pasang semua "alat dan bahan" yang dibutuhin PakDe:
    \`\`\`bash
    pip install flask tensorflow pillow numpy
    \`\`\`
    Kalau TensorFlow-nya susah, coba versi CPU aja:
    \`\`\`bash
    pip install flask tensorflow-cpu pillow numpy
    \`\`\`

6.  Jalanin deh aplikasinya:
    \`\`\`bash
    python app.py
    \`\`\`

7.  Buka browser kesayanganmu 🌐 dan ketik alamat ini:
    \`\`\`
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    \`\`\`
    atau
    \`\`\`
    http://localhost:5000/
    \`\`\`

### ⚠️ **Penting Diingat\!**

  - Pastiin file "otaknya" PakDe (`pakde_folder_model.keras`) ada di folder yang sama kayak file `app.py` ya\! 😉
  - PakDe ini picky soal gambar, maunya format JPEG atau PNG aja 😉.
  - Gambar yang kamu upload bakal diubah dulu ukurannya jadi 224x224 piksel biar otaknya PakDe gampang mikirnya 📐.

## ⚙️ **Cara Kerja PakDe dalam Sekejap:**

1.  Kamu **upload** gambar tulisan tanganmu 📤 (spiral, meander, atau gelombang).
2.  Otak **AI** PakDe langsung **menganalisis** polanya 👀 buat ngedeteksi kemungkinan Parkinson.
3.  Kamu langsung dapet **hasil deteksi** lengkap dengan tingkat kepercayaannya\! ✅ Ini bisa jadi gambaran awal buat kamu.

## 🦸‍♂️🦸‍♀️ **Tim Super di Balik PakDe:**

  - **Tengku Ryan Adiansyah Bani** - Si Jago Deployment 🚀
  - **Reisya Junita Putri** - Ahli Data 📊
  - **Vania Rachmawati Dewi** - Master AI 🧠
  - **Kenny Aldebaran Roberts** - Kapten Proyek 👨‍

## ❗ **Ingat Baik-Baik ya\!**

Hasil dari PakDe ini cuma **indikasi awal** aja ya\! **Bukan berarti ini diagnosis resmi dari dokter\!** 🧑‍⚕️ Tetep konsultasi sama dokter beneran buat evaluasi yang lebih lengkap dan akurat. Kesehatan itu nomor satu\! 👍

## 📞 **Hubungi Kami:**

  - **Email**: cintapakde111@gmail.com 📧
  - **Telepon**: +62 111 222 333 📞

-----

© 2025 PakDe - Parkinson Detection. All rights reserved - imnayr. 😎


"# Website-PakDe" 
"# Website" 
