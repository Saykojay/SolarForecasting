# Panduan Menjalankan Sistem di RunPod (Cloud GPU)

## Persiapan Awal
1. Buat akun di [RunPod.io](https://www.runpod.io/).
2. Isi saldo (*Add Funds*) menggunakan kartu kredit, debit jenius/jago, atau fitur pembayaran lain yang tersedia, minimal $10. Tarifnya sangat murah per jam sehingga saldo akan tahan berhari-hari.

## Langkah 1: Deploy Pod
1. Masuk ke **Menu Pods**.
2. Klik tombol **Deploy**.
3. Pilih tipe GPU yang diinginkan. Rekomendasinya:
   * **RTX 3090 / 4090** (Paling cocok dan hemat untuk pipeline Anda).
   * Atau A4000 jika sekadar untuk tuning.
4. Pilih **"Secure Cloud"** karena koneksinya lebih stabil dari Community Cloud.
5. Pada bagian **Template**, Anda bisa membiarkannya pada default (`RunPod TensorFlow 2.x ...`). Sangat disarankan untuk memilih template **Jupyter TensorFlow**.
6. Klik **Deploy On-Demand**, lalu tunggu hingga status Pod berubah menjadi `Running`.

## Langkah 2: Mengakses Pod & Menyalin File
1. Setelah status `Running`, klik tombol **Connect** pada antarmuka Pod Anda.
2. Anda akan disuguhkan opsi koneksi:
   * Klik **Connect to Jupyter Lab**. (Lebih stabil untuk memantau server dengan antarmuka grafis terminal).
3. Di dalam Jupyter Lab, buka **Terminal Baru** (`File > New > Terminal`).
4. **Opsional tapi Krusial:** Karena file project `Modular Pipeline v1` cukup besar, yang paling efisien adalah menggunakan Git:
   ```bash
   # Di dalam terminal RunPod
   cd /workspace
   git clone [MASUKKAN URL REPOSITORI GITHUB ANDA DI SINI]
   
   # Atau jika Anda mem-zip folder dari komputer, Anda dapat
   # menggunakan fitur 'Upload Files' Jupyter Lab di sebelah kiri, 
   # lalu unzip menggunakan perintah:
   # unzip Modular_Pipeline_v1.zip
   ```
5. Masuk ke folder proyek Anda:
   ```bash
   cd "Modular Pipeline v1" # Atau nama folder repo GitHub Anda
   ```

## Langkah 3: Menginstal Dependensi (Python Library)
RunPod sudah memiliki Conda dan PIP terinstal.
1. Jalankan baris berikut di Terminal Jupyter Lab tadi:
   ```bash
   pip install -r requirements.txt
   ```
   *(Tunggu beberapa menit hingga proses download seluruh lib seperti Tensorflow, Optuna, dkk selesai).*

## Langkah 4: Expose / Buka Port untuk Mengakses Streamlit
Streamlit berjalan secara default di port `8501`. Kita harus mengarahkan koneksi RunPod agar *Traffic HTTP* di 8501 terekspos ke Anda.

1. Kembali ke Dashboard RunPod (di browser luar).
2. Di card (detail) Pod yang sedang berjalan, klik panah untuk melihat opsi **Edit Pod** atau tanda "Settings".
3. Temukan pengaturan **TCP Port Expose** atau **Symmetric Port Forwarding**.
4. Tambahkan/buka port `8501` ke eksposure publik.

## Langkah 5: Menjalankan Dashboard
1. Kembali ke Terminal Jupyter Lab Anda.
2. Jalankan perintah dewa untuk menjalankan app Streamlit Anda di background tanpa terganggu saat terminal Jupyter ditutup secara tidak sengaja:
   ```bash
   nohup streamlit run app.py > streamlit.log 2>&1 &
   ```
   *Atau jika ingin melihat tampilannya langsung (mudah ditutup/ctrl+c)*:
   ```bash
   streamlit run app.py
   ```
3. Terminal akan memunculkan URL internal dan ekstrenal.

## Langkah 6: Mengakses Dashboard Anda dari Komputer (Windows)
1. Pergi ke Dashboard **RunPod**.
2. Klik tombol **Connect**. 
3. Di sebelah tombol Web Terminal / Jupyter Lab, seharusnya Anda akan mendapati tombol **"Connect to HTTP Port 8501"**. 
4. Klik tautan tersebut, maka ia akan mengarahkan browser Anda menuju antarmuka Visual Streamlit Pipeline layaknya dari `localhost`, **TAPI KINI BERJALAN DI SERVER RTX**.

## Langkah 7 (PENTING): Mematikan & Menyimpan Biaya
Biaya RunPod akan **terus berjalan ($/jam)** selama Pod dalam status `Running`, TAPI ia **tetap menyimpan tarif penyimpanan SSD/Storage** kecil meskipun Pod berstatus `Stopped`.
* Jika eksperimen Optuna Tuning sudah selesai selama berjam-jam:
   1. Unduh (download) folder `models/`, `configs/config.yaml`, atau `data/processed/` dari file manager Jupyter Lab RunPod kembali ke komputer / laptop Windows (Lenovo) Anda. Lakukan evaluasinya di laptop Anda secara gratis.
   2. Hentikan Pod untuk menghentikan argometer per jam tinggi (**Stop Pod**).
   3. Jika yakin besok tidak akan melatih mesin lagi, Anda **HARUS KLIK TERMINATE POD (Delete)** agar tidak dikenakan tarif Storage harian secara pasif.
