# Rencana Metodologi Penelitian
## Studi Komparasi Konfigurasi Sistem *Power-to-Methanol*: Analisis Tekno-Ekonomi Berbasis Variasi Topologi Komponen dan Koneksi Jaringan Listrik (*On-Grid* vs *Off-Grid*)



---

## Latar Belakang & Justifikasi Penelitian

Pemilihan konfigurasi koneksi jaringan listrik (*on-grid* atau *off-grid*) merupakan keputusan desain fundamental yang memiliki implikasi signifikan terhadap keekonomian dan keandalan sistem *Power-to-Methanol* (PtM). Selain itu, pemilihan **topologi komponen** — yaitu kombinasi perangkat penyimpanan energi yang digunakan (baterai, tangki hidrogen, elektroliser, atau kombinasinya) — turut menentukan kinerja teknis dan nilai LCOM sistem secara keseluruhan.

Penelitian ini bertujuan untuk **membandingkan berbagai konfigurasi sistem secara sistematis** melalui variasi dua dimensi utama:
1. **Dimensi Koneksi:** *Off-Grid* (100% mandiri) vs *On-Grid* (dengan bantuan jaringan listrik)
2. **Dimensi Topologi Komponen:** Kombinasi keberadaan dan kapasitas Baterai, Tangki H₂, dan Elektroliser PEM

Analisis ini akan mengidentifikasi konfigurasi optimal yang meminimalkan *Levelized Cost of Methanol* (LCOM) untuk kondisi iklim tropis NTT, sekaligus memberikan pemahaman tentang peran masing-masing komponen dalam menjamin kontinuitas produksi.

## Kesenjangan Penelitian (*Research Gap*) & Kebaruan

Berdasarkan identifikasi literatur terkini (2020–2024), penelitian ini didesain untuk menjembatani empat kesenjangan penelitian (*research gap*) utama dalam studi perancangan sistem *Power-to-Methanol* (PtM):

1. **Ketiadaan Kerangka Integrasi Tiga Domain:** Belum ada studi terverifikasi yang mengintegrasikan pembangkitan profil kinerja PV berbasis *Machine Learning*, optimasi tekno-ekonomi *sizing* menggunakan HOMER Pro, dan simulasi sintesis metanol menggunakan Aspen Plus secara *end-to-end* dalam satu kerangka kerja sekuensial yang utuh.
2. **Keterputusan Rantai Data Pokok (*Renewable-to-Process*):** Mayoritas kajian PtM eksisting hanya mengoptimasi sisi suplai energi saja atau sisi proses konversi kimia saja. Penelitian ini menyuguhkan inovasi aliran data bersama (*shared data flow*) linier dari hulu (prakiraan iklim NTT) hingga ke hilir (reaktor kimia Aspen).
3. **Paradigma Komparasi Klasik:** Meskipun komparasi topologi energi (*On-Grid* vs *Off-Grid*) sudah kerap dibahas, namun analisis komparatif ini sama sekali belum pernah diuji menggunakan kombinasi metodologi optimasi *pipeline* HOMER-ke-Aspen yang disetir secara absolut oleh pemodelan prediktif kecerdasan buatan (*Deep Learning-driven pipeline*).
4. **Manipulasi Data Surya (Injeksi *Custom Profile*):** Penggunaan profil *synthetic/custom* sebagai rekayasa input utama (*Custom Production Profile*) ke dalam mesin komputasi HOMER — menggantikan konvensi penggunaan data stasiun satelit cuaca yang kaku — masih tersisa sebagai pendekatan kebaruan (*underexplored angle*) untuk mengatasi volatilitas variabel *Renewable Energy Source* (RES) dalam sistem PtM berskala industri.


## Ringkasan Metodologi

| Aspek | Keterangan |
| :--- | :--- |
| **Pendekatan** | *Sequential Hybrid*: AI Forecasting → HOMER Pro → Aspen Plus |
| **Dimensi Komparasi 1** | Koneksi Jaringan: *Off-Grid* vs *On-Grid* |
| **Dimensi Komparasi 2** | Topologi Komponen: variasi keberadaan Baterai, Tangki H₂, Elektroliser |
| **Variabel Terikat** | LCOM ($/ton), NPC ($), LCOH ($/kg H₂), *Unmet Load* (%) |
| **Variabel Lingkungan** | Emisi CO₂ Operasional (ton/tahun) — berbasis konsumsi grid PLN per skenario *On-Grid* |
| **Target Produksi** | 60.000 ton metanol/tahun (kontinu, kemurnian ≥ 99,85% mol) |
| **Lokasi Studi** | Nusa Tenggara Timur (−10,297°LS; 123,591°BT) |
| **Model Prakiraan** | GRU deterministik — profil 8.760 jam/tahun |
| **Perangkat Optimasi** | HOMER Pro (minimasi NPC, *exhaustive search*) |
| **Perangkat Simulasi** | Aspen Plus (*steady-state*, SRK & NRTL) |

---

## Matriks Skenario Penelitian

Berikut adalah usulan matriks skenario yang mencakup variasi topologi dan koneksi jaringan. Skenario bersifat **tidak terbatas** dan dapat diperluas atau dimodifikasi sesuai arahan pembimbing.

| No. | Label Topologi Sistem | Koneksi Jaringan | Pembangkit Eksisting | Penyimpanan Bawaan | Keterangan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **PV** (*Baseline Off-Grid*) | *Off-Grid* | PV Saja | ❌ (Tanpa Penyangga) | Baseline skenario mandiri murni tanpa unit tambahan penyimpanan |
| 2 | **PV - Battery** | *Off-Grid* | PV Saja | Baterai Saja | Sistem mandiri dengan penyangga kapasitas listrik |
| 3 | **PV - H2 Tank** | *Off-Grid* | PV Saja | Tangki H₂ Saja | Sistem mandiri yang mengandalkan fleksibilitas operasi elektroliser dan gas buffer |
| 4 | **Grid - PV - Battery** | *On-Grid* | Grid + PV | Baterai Saja | Cadangan grid hibrida dengan kontrol luahan baterai internal |
| 5 | **Grid - PV - Battery - H2 Tank**| *On-Grid* | Grid + PV | Baterai + Tangki H₂ | Kombinasi operasional terlengkap (investasi awal tertinggi) |
| 6 | **Grid - PV - H2 Tank** | *On-Grid* | Grid + PV | Tangki H₂ Saja | Kestabilan dibebankan pada variabilitas tarif harga grid pelengkap |
| 7 | **Grid - PV** (*Baseline On-Grid*) | *On-Grid* | Grid + PV | ❌ (Tanpa Penyangga) | Baseline studi konfigurasi dengan cadangan aliran energi dari jaringan terpusat |

> [!NOTE]
> Matriks di atas adalah *starting point* yang dapat dikembangkan. Misalnya, setiap skenario dapat dijalankan dengan beberapa rentang kapasitas (*sizing sweep*) untuk mengidentifikasi titik optimal masing-masing konfigurasi. Skenario tambahan dapat diusulkan berdasarkan diskusi dengan pembimbing.

> [!NOTE]
> **Analisis Sensitivitas Asal Grid:** Selain variasi metrik topologi di atas, skema evaluasi juga akan mencakup sensitivitas terhadap **harga dan intensitas emisi karbon dari pasokan listrik jaringan** (contoh: kondisi grid defisit tinggi vs. grid terdekarbonisasi) guna menilai kelayakan tekno-ekonomi jangka panjang operasional *On-Grid*.
> [!IMPORTANT]
> Kontribusi ilmiah utama penelitian ini adalah peta komparasi LCOM dari seluruh skenario di atas, yang memberikan **panduan desain berbasis data** mengenai konfigurasi sistem PtM yang paling ekonomis untuk kondisi spesifik iklim tropis NTT.

---

## Kerangka Metodologi: *Sequential Hybrid Optimization*

Metodologi terdiri atas tiga tahap yang dikerjakan secara berurutan, di mana keluaran (*output*) setiap tahap menjadi masukan (*input*) bagi tahap berikutnya.

```
Tahap I                    Tahap II                   Tahap III
─────────────────          ─────────────────          ─────────────────
Pemodelan Prakiraan   →    Optimasi Sistem        →   Simulasi Proses
Daya Surya                 Mikrogrid                  Kimia
(Python / GRU)             (HOMER Pro)                (Aspen Plus)
Output: Profil PV          Output: Sizing             Output: LCOM &
8760 jam (CSV)             Optimal Komponen           Verifikasi Yield
                                ↓
                      Tahap IV: Analisis Tekno-Ekonomi & Lingkungan
                      ──────────────────────────────────────────
Kalkulasi LCOM final + Emisi CO₂ Grid per Skenario On-Grid
```

---

## Tahap I — Pemodelan Prakiraan Daya Surya

### 1.1 Tujuan
Menghasilkan profil daya energi surya per jam selama satu tahun operasional (8.760 data poin) yang merepresentasikan **Typical Meteorological Year (TMY)** di lokasi studi NTT menggunakan model *Deep Learning*. Tujuan pemodelan ini bukan untuk meramal kondisi cuaca presisi pada tanggal di tahun masa depan secara spesifik, melainkan membangkitkan sebuah profil representatif sintetik yang mengekstrak dinamika ekstrem serta tren iklim historis 10 tahun terakhir langsung menjadi representasi fluktuasi **Normalized Power Output** (*Capacity Factor*).

### 1.2 Spesifikasi Model

| Parameter | Keterangan |
| :--- | :--- |
| **Arsitektur** | *Gated Recurrent Unit* (GRU) |
| **Fitur Masukan** | DHI (W/m²), DNI (W/m²), Suhu Udara (°C), *hour_cos*, *hour_sin* |
| **Target Output** | Daya Output Ternormalisasi / *Capacity Factor* (0–1) per jam |
| **Sumber Data** | Data historis 10 tahun, lokasi NTT (koordinat −10,297°LS; 123,591°BT) |
| **Resolusi Temporal** | 1 jam (resolusi sinkron dengan model optimasi) |
| **Pembagian Data** | 70% pelatihan / 30% pengujian |
| **Optimasi Hiperparameter** | *Bayesian Search* menggunakan Optuna |

### 1.3 Prosedur End-to-End

**a. Pengumpulan, Pra-pemrosesan Data & Pembangkitan TMY Sintetik**
- Pengunduhan data meteorologi dari NASA POWER atau PVGIS untuk koordinat lokasi studi (data historis 10 tahun).
- Rekayasa profil masukan masa depan (*Future Input Profiling*): Merangkum seluruh variabel masukan (Suhu, DNI, DHI) ke dalam satu profil *Typical Meteorological Representative* sebagai basis input model.
- Normalisasi fitur menggunakan *MinMaxScaler*.
- Pembentukan urutan masukan (*lookback window*) dengan panjang 24 jam.
- Pemisahan data latih dan uji (80:20).

**b. Perancangan Arsitektur GRU**
- Susunan 2–3 lapisan GRU (*stacked*) dengan unit tersembunyi yang dioptimasi Optuna
- Lapisan *Dropout* untuk regularisasi dan pencegahan *overfitting*
- Lapisan keluaran *Dense* dengan 1 neuron dan fungsi aktivasi linear

**c. Pelatihan Model**
- *Loss function*: *Mean Squared Error* (MSE)
- *Optimizer*: Adam dengan *learning rate scheduler*
- *Early stopping* untuk mencegah pelatihan berlebih

**d. Evaluasi Kinerja Model**
- Metrik: *Root Mean Squared Error* (RMSE), *Mean Absolute Error* (MAE), koefisien determinasi (R²)
- Validasi visual: perbandingan kurva prediksi vs aktual pada hari tipikal

**e. Eksekusi Prediksi (Proyeksi TMY) & Ekspor Keluaran untuk HOMER**
- Model memetakan TMY sintetik langsung ke nilai daya output ternormalisasi (0 hingga 1) per jam. Pendekatan ini mewakili *Capacity Factor* temporal dari respons sistem PV terhadap iklim NTT.
- Simpan profil prediksi sepanjang 8.760 jam ke dalam format `.txt` atau `.csv`.
- Format file disesuaikan dengan kebutuhan impor **Custom Production Profile** di HOMER Pro (berisi entri data kolom tunggal nilai fluktuasi daya 0–1 secara berurutan).
- Tidak boleh ada nilai kosong (NaN) → celah data pada profil dasar TMY wajib diisi via interpolasi linier agar simulasi tahunan berkesinambungan.

---

## Tahap II — Optimasi Perancangan Sistem Mikrogrid

### 2.1 Tujuan
Menentukan kapasitas optimal komponen sistem energi (*sizing*) yang meminimalkan *Net Present Cost* (NPC) dengan tetap menjamin pemenuhan target produksi hidrogen secara kontinu.

### 2.2 Perangkat Lunak & Komponen

HOMER Pro melakukan pencarian optimal secara *exhaustive* atas semua kombinasi kapasitas yang didefinisikan pengguna, mensimulasikan keseluruhan operasi 8.760 jam, kemudian memilih konfigurasi dengan NPC terendah.

| Parameter | Keterangan |
| :--- | :--- |
| **Perangkat Lunak** | HOMER Pro |
| **Sumber Energi** | *Solar PV* (profil daya ternormalisasi tahap I diimpor ke HOMER sebagai *Custom Production Profile*) |
| **Komponen Penyimpanan** | Baterai *Lithium-ion*, Tangki Penyimpanan Hidrogen |
| **Komponen Konversi** | Elektroliser PEM (*built-in HOMER module*) |
| **Fungsi Tujuan** | Minimasi *Net Present Cost* (NPC) |
| **Konstrain Utama** | *Hydrogen Load* terpenuhi sepanjang tahun (tanpa *unmet load*) |

### 2.3 Derivasi Target Produksi Hidrogen

Beban hidrogen (*Hydrogen Load*) yang dimasukkan ke HOMER diturunkan dari target produksi metanol melalui stoikiometri reaksi sintesis:

**Reaksi sintesis metanol:** CO₂ + 3H₂ → CH₃OH + H₂O

| Parameter | Nilai |
| :--- | :--- |
| Target produksi metanol | 60.000 ton/tahun |
| Rasio massa H₂ terhadap metanol | 6/32 = 0,1875 |
| Kebutuhan H₂ teoritis | ±11.250 ton H₂/tahun |
| Kebutuhan H₂ aktual (efisiensi konversi ~90%) | ±12.500 ton H₂/tahun |
| **Beban H₂ harian** | **±34.247 kg H₂/hari** |
| **Beban H₂ per jam (profil datar)** | **±1.427 kg H₂/jam** |

Dalam pemodelan HOMER, target produksi metanol direpresentasikan sebagai ***Hydrogen Flat Load*** sebesar 1.427 kg/jam sepanjang 8.760 jam, mengingat HOMER tidak memiliki modul sintesis metanol.

### 2.4 Ruang Pencarian Variabel Keputusan

| Komponen | Rentang Kandidat | Satuan |
| :--- | :--- | :--- |
| Kapasitas *Solar PV* | 50 — 500 | MW |
| Kapasitas Baterai *Lithium-ion* | 50 — 500 | MWh |
| Kapasitas Elektroliser PEM | 50 — 200 | MW |
| Volume Tangki Penyimpanan H₂ | 10.000 — 100.000 | kg |

### 2.5 Keluaran Optimasi

- Konfigurasi kapasitas optimal untuk setiap komponen
- Profil operasional per jam: daya PV, laju produksi H₂, *State of Charge* baterai
- Rincian biaya ekonomi: CAPEX, OPEX, NPC, dan *Levelized Cost of Hydrogen* (LCOH)

> [!WARNING]
> HOMER Pro tidak mencakup simulasi unit proses kimia (reaktor metanol dan kolom distilasi). Oleh karena itu, nilai LCOM akhir tidak dapat diperoleh langsung dari HOMER, melainkan harus dikalkulasi secara terpisah dengan menggabungkan keluaran ekonomi HOMER dan hasil simulasi Aspen Plus (Tahap III).

---

## Tahap III — Simulasi Proses Kimia

### 3.1 Tujuan
Memvalidasi keterlaksanaan konversi hidrogen menjadi metanol sesuai kapasitas yang ditetapkan HOMER, serta menghitung kontribusi biaya operasional kimia sebagai komponen OPEX dalam formula LCOM.

> [!IMPORTANT]
> **Catatan Penting — Status Input H₂ per Skenario:** Untuk skenario yang **memiliki Tangki H₂** (skenario 3, 5, 6), H₂ keluar dari tangki ke reaktor dalam kondisi stabil karena diratakan oleh *buffer*. Namun untuk skenario **tanpa Tangki H₂** (skenario 1, 2, 4, 7), H₂ mengalir **langsung dan dinamis** dari elektroliser — mengikuti fluktuasi daya PV — tanpa ada penyangga. Pada Aspen Plus, kedua kondisi ini tetap diperlakukan dengan nilai rata-rata (*quasi-steady-state*), namun skenario tanpa tangki akan mencatat *Unmet Hydrogen Load* yang lebih tinggi di HOMER sebagai penanda risiko operasional.

### 3.2 Alur Integrasi (Aspen Plus ↔ HOMER)

Karena dependensi siklis antara konsumsi listrik pabrik (input HOMER) dan kapasitas H₂ dari HOMER (input Aspen), pendekatan iteratif tiga langkah diterapkan:

**Langkah 1 — Estimasi Awal (Aspen → HOMER):**
Simulasi Aspen dijalankan dengan asumsi laju alir H₂ awal berdasarkan estimasi kapasitas pabrik. Total konsumsi listrik seluruh unit operasi (kompresor, pompa, utilitas) dicatat dan dikonversi menjadi profil beban listrik (*Electric Load*) untuk HOMER.

**Langkah 2 — Optimasi Sistem (HOMER):**
HOMER menentukan konfigurasi PV, Baterai, dan Elektroliser optimal berdasarkan profil beban listrik dan target H₂. Keluaran utama: profil produksi H₂ per jam dan rata-rata laju harian.

**Langkah 3 — Validasi & Kalkulasi Final (HOMER → Aspen):**
Laju alir H₂ hasil HOMER dimasukkan ke Aspen Plus untuk memverifikasi bahwa produksi metanol memenuhi target ≥ 60.000 ton/tahun dan kemurnian produk ≥ 99,85% mol. Biaya operasional kimia (katalis, air proses, utilitas) dikalkulasi sebagai komponen OPEX.

---

### 3.2.1 Contoh Keluaran HOMER yang Diteruskan ke Aspen Plus

HOMER menghasilkan laporan lengkap dalam format tabel dan grafik. Data kunci yang diekspor untuk keperluan simulasi Aspen Plus adalah sebagai berikut:

| Data Keluaran HOMER | Contoh Nilai | Keterangan |
| :--- | :--- | :--- |
| Kapasitas Elektroliser Terpilih | 120 MW | Kapasitas desain PEM yang dioptimalkan |
| Rata-rata Produksi H₂ Harian | 32.400 kg/hari | Basis laju alir untuk simulasi *steady-state* |
| Rata-rata Produksi H₂ Per Jam | 1.350 kg/jam | Nilai ini menjadi *feed flowrate* di Aspen |
| *Capacity Factor* Elektroliser | 84% | Fraksi waktu elektroliser beroperasi |
| CAPEX Sistem Energi | $210 juta | Dimasukkan ke kalkulasi NPC dan LCOM |
| OPEX Tahunan Sistem Energi | $8,5 juta/tahun | Termasuk penggantian stack PEM |

> [!NOTE]
> Nilai rata-rata tahunan digunakan sebagai basis simulasi *steady-state* Aspen Plus. Pendekatan ini valid secara akademis karena Aspen Plus mensimulasikan kondisi operasi desain nominal, bukan profil dinamis per jam.

---

### 3.2.2 Perubahan pada Simulasi Aspen Plus Berdasarkan Data HOMER

Simulasi Aspen Plus yang sudah ada (`METHANOL SYNTHESIS HEN REVISI1.apw`) mencakup tiga unit utama: **unit kompresi**, **unit reaksi**, dan **unit distilasi**. Parameter yang **diubah** untuk setiap skenario HOMER adalah:

| Unit Operasi | Parameter yang Diubah | Sumber Nilai |
| :--- | :--- | :--- |
| **Stream Feed H₂** | Laju alir molar H₂ (kmol/jam) | Rata-rata produksi H₂/jam dari HOMER |
| **Stream Feed CO₂** | Laju alir molar CO₂ (kmol/jam) | Disesuaikan secara stoikiometris terhadap H₂ (rasio 1:3) |
| **Unit Kompresi** | Tidak diubah | Tekanan umpan dan spesifikasi kompresor tetap sama |
| **Unit Reaktor (RPlug)** | Tidak diubah | Kondisi operasi reaktor (T, P, katalis) tetap |
| **Unit Distilasi** | Tidak diubah | Spesifikasi kemurnian produk tetap 99,85% mol |

**Pendekatan simulasi: *Steady-State*.**
H₂ yang masuk ke Aspen Plus diperlakukan sebagai aliran kontinu dengan laju tetap (nilai rata-rata dari HOMER). Aspen Plus tidak memodelkan fluktuasi jam ke jam, melainkan mensimulasikan kondisi operasi desain nominal. Hal ini konsisten dengan pendekatan *quasi-steady state* yang umum digunakan dalam studi tekno-ekonomi sistem PtM (Mucci et al., 2023; Vo et al., 2025).

**Konsep Lapisan Operasi Sistem (Dinamis → Statis):**

Tangki H₂ berfungsi sebagai **penyangga** (*buffer*) yang memisahkan dunia energi yang fluktuatif (HOMER) dengan dunia proses kimia yang memerlukan kondisi stabil (Aspen Plus):

| Lapisan Sistem | Sifat Operasi | Disimulasikan di |
| :--- | :--- | :--- |
| Daya masuk ke Elektroliser (dari PV) | **Dinamis** — mengikuti profil iradiasi per jam | HOMER Pro |
| Produksi H₂ dari Elektroliser | **Dinamis** — proporsional terhadap daya PV | HOMER Pro |
| H₂ keluar dari Tangki menuju pabrik | **Terstabilkan** — diratakan oleh buffer tangki | Nilai rata-rata diambil dari HOMER |
| Umpan H₂ masuk ke Reaktor Aspen | **Steady-State** — kondisi operasi desain nominal | Aspen Plus |

> [!NOTE]
> Pada skenario **tanpa Tangki H₂** (skenario 1, 2, 4, 7), H₂ mengalir langsung dari elektroliser ke reaktor tanpa *buffer*. Asumsi *steady-state* tetap diterapkan di Aspen Plus menggunakan nilai rata-rata H₂ dari HOMER, namun dengan konsekuensi *Unmet Hydrogen Load* yang lebih tinggi.

> [!CAUTION]
> **Pertimbangan Simulasi Dinamis:** Pendekatan yang lebih akurat secara teknis untuk skenario tanpa tangki adalah menggunakan **Aspen Plus Dynamics** (modul terpisah dari Aspen Plus statis), yang mampu memodelkan fluktuasi laju alir H₂ per jam secara penuh. Implikasinya terhadap metodologi adalah sebagai berikut:
> - **Kelebihan:** Hasil simulasi lebih realistis — reaktor terkena *ramp-up/ramp-down* H₂ yang sesungguhnya, sehingga profil konversi dan kemurnian produk lebih akurat.
> - **Kelemahan:** Aspen Plus Dynamics membutuhkan *control loop* (PID controller), spesifikasi *hold-up* tangki reaktor, dan waktu komputasi yang jauh lebih panjang. Kurva belajar lebih curam dan berada di luar cakupan studi *steady-state* pada level sarjana.
> - **Keputusan Metodologis:** Penelitian ini mempertahankan pendekatan *steady-state* untuk seluruh skenario demi konsistensi perbandingan. Keterbatasan ini didokumentasikan secara eksplisit sebagai ruang pengembangan pada riset lanjutan (*future work*).


| Parameter | Keterangan |
| :--- | :--- |
| **Perangkat Lunak** | Aspen Plus |
| **Paket Termodinamika** | SRK (unit reaksi), NRTL (unit distilasi) |
| **Model Reaktor** | RPlug dengan katalis Cu/ZnO/Al₂O₃ |
| **Target Kemurnian Produk** | ≥ 99,85% mol metanol |
| **Unit Operasi** | Kompresor bertahap, Reaktor Sintesis, Menara Distilasi |
| **Pendekatan Simulasi** | *Steady-state* (berdasarkan rata-rata laju H₂ dari HOMER) |

---

## Tahap IV — Analisis Tekno-Ekonomi & Evaluasi Lingkungan

### 4.1 Formulasi LCOM

Nilai *Levelized Cost of Methanol* (LCOM) dikalkulasi dengan menggabungkan keluaran ekonomi dari kedua tahap komputasi:

**LCOM ($/ton) = [ NPC_HOMER + OPEX_Aspen ] / Total Produksi Metanol (ton)**

### 4.2 Metode Analisis

| Metode | Perangkat | Tujuan |
| :--- | :--- | :--- |
| Analisis Sensitivitas | HOMER Pro (*built-in*) | Menguji pengaruh variasi harga komponen dan iradiasi surya terhadap NPC |
| Komparasi Data AI vs. Standar | Python + HOMER | Mengkuantifikasi peningkatan akurasi optimasi akibat penggunaan profil GRU kustom dibandingkan data NASA generik |
| Kalkulasi LCOM | Python / Spreadsheet | Integrasi komponen biaya dari HOMER dan Aspen Plus |

### 4.3 Analisis Emisi Karbon (Dampak Lingkungan)

Pada skenario yang melibatkan koneksi jaringan (*On-Grid*), penggunaan listrik tambahan dari sistem interkoneksi PLN (*Grid Purchases*) akan menyumbang emisi gas rumah kaca tidak langsung (Cakupan 2) pada siklus produksi e-Methanol. Analisis kelayakan lingkungan dievaluasi dengan perhitungan emisi karbon dari komponen eksternal tersebut:

**Emisi CO₂ Operasional (ton/tahun) = Pembelian Grid PLN (kWh/tahun) × Faktor Emisi Grid (kg CO₂/kWh) / 1000**

- **Pembelian Grid PLN:** Diperoleh langsung dari nilai *Grid Purchases* pada laporan optimasi HOMER Pro per skenario *On-Grid*.
- **Faktor Emisi Grid (*Emission Factor*):** Merujuk pada nilai historis/proyeksi faktor emisi sistem ketenagalistrikan regional Nusa Tenggara Timur (berdasarkan publikasi Direktorat Jenderal Ketenagalistrikan Kementerian ESDM).

Evaluasi ini nantinya akan dikomparasikan secara langsung antarskenario untuk memperlihatkan tarik-ulur (*trade-off*) antara penghematan biaya versus status "kadar kehijauan" metanol yang dihasilkan (*green vs. e-methanol*).

---

## Batasan Metodologi

> [!CAUTION]
> Beberapa asumsi penyederhanaan diterapkan dalam metodologi ini yang perlu didokumentasikan secara eksplisit dalam naskah skripsi:
>
> 1. **Optimasi bersifat deterministik.** HOMER Pro tidak menerapkan pemrograman stokastik; variabilitas cuaca diwakili oleh profil prediksi GRU tunggal, bukan distribusi probabilistik. Pendekatan berbasis ketidakpastian stokastik dapat dikembangkan pada riset lanjutan.
>
> 2. **Unit proses kimia tidak terintegrasi dalam *solver* optimasi.** Keterkaitan antara HOMER dan Aspen Plus diselesaikan secara iteratif, bukan secara simultan dalam satu kerangka matematis terpadu seperti pada pendekatan MINLP.
>
> 3. **Elektroliser dimodelkan sebagai komponen sederhana.** Parameter elektroliser menggunakan efisiensi rata-rata dari literatur, bukan berdasarkan simulasi elektrokimia *rigorous*.
>
> 4. **Peniadaan Pemodelan Unit Penangkapan Karbon.** Penelitian ini sangat berfokus pada optimasi fluktuasi energi terbarukan, sehingga tidak memperhitungkan fisik instalasi *Carbon Capture*. Suplai CO₂ diasumsikan tersedia bebas dari sumber eksternal dengan *flowrate* parameter yang akan otomatis beradaptasi (mengimbangi rasio 1:3) terhadap laju suplai H₂ dari elektroliser.

---

## Indikator Keberhasilan Penelitian

| Indikator | Target |
| :--- | :--- |
| Akurasi model GRU | R² ≥ 0,85 (hasil aktual: 0,87 dari data PVOutput.org) |
| Pemenuhan beban H₂ di HOMER | *Unmet Load* = 0% |
| Produksi metanol tahunan | ≥ 60.000 ton/tahun |
| Kemurnian produk metanol | ≥ 99,85% mol |
