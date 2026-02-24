# Strategi Search Space Optuna TPE untuk Model GRU

Membangun *Search Space* untuk algoritma klasik seperti **Gated Recurrent Unit (GRU)** jauh lebih konservatif dan "primitif" dibanding ras raksasa seperti PatchTST atau TimeTracker MoE. GRU memang diciptakan sebagai versi "langsing" dan "ngebut" dari LSTM karena ia menghemat *memory cell gate* di dalam perhitungannya. 

Berdasarkan *best practice* para peneliti *Machine Learning* untuk prediksi deret waktu tabular (cuaca/energi) dengan *budget* TPE raksasa **100 Trials GPU**, berikut adalah parameter eksploitasi terbaik yang memaksa GRU menjadi *baseline* bertenaga super mutan:

---

## 🎯 Tabel Rekomendasi Search Space (GRU dengan 100 Trials)

| Parameter (Hyperparameter) | Rentang Pilihan | Skala / Distribusi | Argumentasi Pakar & Alasan Matematis |
| :--- | :--- | :--- | :--- |
| **`d_model`** *(Hidden Units)* <br>*(Kapasitas Memori Cell)* | **`64` - `256`** | *Categorical* | GRU cuma punya *Update Gate* dan *Reset Gate* (minus *Output Gate* punya LSTM). Jadi ia lebih irit VRAM GPU. Kita bisa memompa ukuran unit `d_model`-nya hingga 256. Lebij dari 256 di data cuaca biasanya akan berujung pada penderitaan *Overfitting* murni (menghafal deret waktu, bukan generalisasi). |
| **`n_layers`** <br>*(Kedalaman Stack/Tumpukan)* | **`1` - `3`** | *Integer* | Ingat hukum RNN/GRU: **Makin dalam, makin bodoh ujungnya!** Karena absennya *Skip/Residual connection* yang paten seperti Transformer, menumpuk lebih dari 3 lapis GRU untuk *Time Series Forecast* dijamin 100% memanen *Vanishing Gradient*. Rentang (1 hingga 3) memberi izin Optuna membuktikan bahwa arsitektur tipis (*1 atau 2 layer*) sering jadi juara di arsitektur rekursif ini. |
| **`lookback`** <br>*(Jendela Toleh Masa Lalu)* | **`24` - `120`** <br>*(1 sampai 5 Hari)* | *Categorical (Step 24)* | Tidak seperti Transformer (*TimeTracker*) yang kebal dan amnesia, GRU/LSTM punya tabiat "cepat pikun" untuk jendela waktu yang ditarik terlalu mundur ke belakang (*Long-range dependency issue*). Menatap ke memori minggu penuh (168 jam) adalah bunuh diri. Rentang 1 sampai 5 hari sangat cukup melatih TPE memilih panjang rentang *Sweet Spot*-nya. |
| **`batch_size`** <br>*(Umpan Per Update)* | **`32`** <br>*(Dikunci/Statis)* | *Fixed* | GRU adalah model sekuensial langkah-demi-langkah (Time-Step by Time-Step) di dalam loop perhitungan GPU-nya. Untuk menghindari isu kebocoran memori (CUDA OOM) dan CUBLAS *execution error* yang fatal saat komputasi panjang, *batch size* secara statis dikunci pada angka 32. Ini menjamin stabilitas sistem 100% selama siklus Optuna. |
| **`learning_rate`** <br>*(Kecepatan Turun Gunung Loss)* | **`0.0005` - `0.005`** | *Log-Uniform* | Bentang geometri turunan (Loss Space/Landscape) dari Arsitektur GRU itu halus tapi kadang gampang tersangkut di bukit palsu (*Local Minima*). Angka LR harus diberi batas wajar (jangan serendah transformer 0.0001) agar model cukup lincah berguling mencari lembah konvergensi yang tepat, terutama bila dipanggang Oplosan *Loss Huber*. |
| **`dropout`** <br>*(Skeptisisme Neouron Acak)* | **`0.1` - `0.3`** | *Float* | Terus terang kapasitas GRU untuk "terjebak hafalan" sangatlah mudah. Namun, memberinya obat Dropout terlalu tinggi (>0.3) bakal merusak kontinuitas sinyal deret waktu. Optuna butuh ruang mencari keseimbangan obat penenang *Overfit* ini. |
| **`use_bidirectional`** <br>*(Melihat Dua Arah)* | **`True`** <br>*(Wajib Nyala)* | *Boolean* | GRU **wajib** membaca cuaca maju (Waktu=T ke Waktu=T+N) dan mundur (Waktu=T+N ke Waktu=T). Kenapa? Karena pergerakan kurva PV Matahari di penghujung fajar dan senja itu adalah cermin simetris. Mematikan fitur *Bidirectional* adalah mengkhianati filosofi deret waktu fisik! |
| **`use_revin`** <br>*(Gunakan Reversible Normalization)* | **`True`** <br>*(Sangat Direkomendasikan)* | *Boolean* | Fitur "Pelindung Anti-Anomali" yang telah ada di Streamlit Anda. Berkat ini, GRU *nggak* akan kaget kalau besok mendadak terjadi gerhana matahari atau badai parah karena datanya sudah dinetralisasi distribusinya (*Distribution Shift Shield*). |

---

## 🔬 Teori Evolusi TPE (100 Trials) dengan Mesin GRU Mungil

Dengan budget "Sultan" sebanyak **100 Trials** mengeksekusi arsitektur primitif seperti GRU: Pesta pora bagi Optuna! Eksekusi 1 model GRU akan terasa *sekejap mata* diabndingkan PatchTST/TimeTracker.

1.  **Iterasi 1 - 25 (Penyapuan Randah):** TPE akan membabi-buta menumpuk (`n_layers` = 3) dan (`lookback` = 120 jam) bersamaan, lalu menyadari dengan takjub kalau *Loss* modelnya hancur berantakan akibat siksaan degradasi panjang LSTM-cell memory.
2.  **Iterasi 26 - 50 (Kesadaran Arsitektur Tipis):** TPE akan "terbangun" bahwa GRU cuma butuh sedikit tumpuk (`n_layers` = 1 atau 2) namun dikompensasi ukuran otak/cell yang rakus (`d_model` = 128 atau 256).
3.  **Iterasi 51 - 100 (Operasi Pisau Bedah Mikroskopik):** Karena topologi *Loss* untuk arsitektur ini sudah ia petakan dengan paripurna per iterasinya, seluruh model (50 model tersisa ini) akan dicoba berkonsentrasi memutar obeng putaran desimal terhalus dari `dropout` (seperti 0.155 ke 0.160) dan `learning_rate` untuk menekan loss error terendah (MSE mendekati titik nihil).

**Pesan Penting Jika Anda Ingin Eksekusi GRU:**  
Latih model GRU (ditambah *check-box* RevIN dan Bidirectional) sebegai fondasi atau pagar pelindung (*Baseline/Benchmark Target*). Kalau **TimeTracker** atau **PatchTST** Anda hasil parameter tuningnya tidak mampu mengalahkan *Loss Score* dari GRU mutan (hasil 100 Trial ini), berarti sang Transformers modern butuh dibongkar celah *Search Space* barunya lebih lanjut. Jangan buang GRU, ini juri paling adil dan beringas!
