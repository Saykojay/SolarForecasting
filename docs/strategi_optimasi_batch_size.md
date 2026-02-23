# Strategi Eskalasi dan Optimasi Pelatihan Melalui Manipulasi Batch Size

Dokumen ini merangkum teknik dan seni memanipulasi ukuran himpunan data (`Batch Size`) untuk mempercepat waktu uji (Tuning) maupun fase pelatihan (Training) model Deep Learning secara drastis, baik di ekosistem peranti keras lokal (Laptop CPU) maupun server *cloud* bertenaga GPU (misal: RunPod).

---

## 🏎️ Strategi 1: Batch Size Scheduling (Percepatan Bertahap)

Murni mengandalkan *batch size* kecil terus-menerus akan menyita waktu eksekusi Anda, sedangkan langsung menggunakan *batch size* raksasa sedari awal bisa membuat akurasi prediksi meleset secara serampangan. 

Solusinya adalah pendekatan dinamik: **Batch Size Scheduling**.

Sistem Pipeline ini secara natif telah terintegrasi dengan modul penjadwalan ini:
1. **Fase Awal (Pencarian Halus):** Menggunakan Batch Size ukuran standar (misal: 32). Jaringan diberikan keleluasaan bergerak mencari arah "*loss valley*" (lembah kegagalan terkecil) dengan akurat.
2. **Fase Pertengahan (Paralelisme Agresif):** Menggandakan batasan (misal: 64). Mempercepat jalannya pengerjaan setelah kompas kebenaran akurasi stabil dan dapat diprediksi model.
3. **Fase Penutup (Fine-Tuning Kilat):** Menggandakan ke ukuran masif (misal: 128/256). Proses pematangan model terakhir (*Global Minima*) kini berjalan sekejap mata menyambar semua kemampuan inti GPU/CPU yang tersedia di akhir perjalanannya.

**Cara Menggunakan:**
* Centang kotak **"📈 Batch Size Scheduling"** di layar Dasbor UI tab **Training Center**.
* *(Hati-hati)*: Tetapkan "Batas Maksimal Batch Size (Limit)" secara sadar. Jangan membatasinya terlalu tinggi melebihi RAM/VRAM yang Anda miliki.

---

## ⚡ Strategi 2: Penggunaan Data Subsampling untuk Tuning Hyperparameter Tertutup

Pada tahap *Tuning Hyperparameter* (mencari racikan ramuan arsitektur model di Optuna), menggunakan seluruh kumpulan himpunan data ratusan ribu jam sangatlah mubazir. Optuna hanya butuh sebuah 'celah perbandingan' untuk memilah kandidat racikan model mana yang lebih akurat, **bukan** belajar menghafal data untuk kelulusan ujian akhirnya.

**Taktik Percepatan 10x Lipat:**
1. Potong data yang diolah Optuna. Gunakan hanya porsi data paling relevan (misal: **20% data** deret waktu yang paling baru).
2. Biarkan Optuna dengan gagah merampungkan 100 *trials* dalam waktu yang dipangkas tajam (misal dari ekspektasi ±20 jam menjadi hanya hitungan kisaran menit/satu jam).
3. Setelah sang pemenang parameter "*The Best*" muncul, kembalilah mengeksekusinya di menu latih (**Training Center**) **TETAPI kali ini** gunakan skala porsi data 100%.

**Cara Menggunakan:**
* Centang **"🚀 Gunakan Taktik 1: Data Subsampling"** di Tab Dasar **Tuning**. 
* Geser porsi slider menjadi **20%** atau **30%**.

---

## 🥊 Strategi 3: Pemilihan "Medan Perang" Paling Ideal: CPU vs GPU

Percaya atau tidak, menggunakan alat *computing* paling mutakhir tidak selalu menjanjikan garansi kehebatan kecepatan. Kenali anatominya:

### A. Kapan Menggunakan CPU Saja? (Mode Laptop/Desktop)
* Bila *Batch Size* yang Anda setel terlampau kecil (misal: **< 16**), memori GPU lebih banyak bengong daripada mengeksekusi perhitungan.
* **CPU lebih handal** meladeni eksekusi rentetan linear karena *clock speed* dan arsitekturnya yang merespons latensi "*nano-detik*" jauh lebih gesit dari GPU yang bergantung pada antrean kabel koneksi.

### B. Kapan Wajib Menyewa GPU Berangasan? (Mode RunPod / Cloud Server)
* Saat model Anda memakai **Batch Size Besar (64, 128, 256, 512)**. CPU Anda sekelas *Core Ultra 7 / i9* pun akan megap-megap kehabisan nafas. GPU seperti A4000 atau RTX 3090 / 4090 akan melahap tumpukan matriks seratusan data itu layaknya sebuah pabrik mesin penggilas aspal —sekali sapuan selesai *parallel vector calculation*.
* Saat durasi target *training* Anda mencapai hitungan belasan jam. Chip komputer kelas konsumen akan melepuh oleh siksaan pelambatan panas *Thermal Throttling*. Pangkalan industri menjaganya pada *Clock Turbo Maksimal* secara stabil seharian penuh.

---

**Kesimpulan Harmonisasi Eksekusi:**
- *Tuning Eksperimental Rumahan:* CPU Lokal + Data Subsampling (20%) + Batch Size Standar (32).
- *Pelatihan Ekstraksi Final (Production Ready):* Sewa Mesin GPU Cloud + Data Penuh (100%) + Batch Size Terjadwal Berkala (Mulai dari 32 ke 256).
