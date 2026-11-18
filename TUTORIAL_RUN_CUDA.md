# ⚡ Tutorial: Menjalankan Baseline Eksperimen dengan Akselerasi CUDA (NVIDIA GeForce RTX 3070 Ti)

> **Repositori Target:** `EXPERIMENT/Code/baseline_gaur/`  
> **Target Hardware:** GPU NVIDIA GeForce RTX 3070 Ti (8GB GDDR6X VRAM, CUDA Compute Capability 8.6)  
> **Environment Manager:** `uv` (Astral)  
> **Framework:** PyTorch 2.x dengan CUDA 12.1 / 11.8 Backend  
> **Dataset Utama:** DEAM (1.802 audio tracks & anotasi statis/dinamis)

---

## 📋 1. Prasyarat Sistem & Driver

Sebelum memulai, pastikan sistem telah memenuhi persyaratan berikut:
1. **NVIDIA Display Driver:** Versi $\ge 530.xx$ (mendukung CUDA 12.x).
2. **Cek Driver GPU di Terminal:**
   ```powershell
   nvidia-smi
   ```
   *Pastikan kartu grafis `NVIDIA GeForce RTX 3070 Ti` terdeteksi beserta versi CUDA Driver.*
3. **Python:** Python 3.10, 3.11, atau 3.12 (disarankan Python 3.11).
4. **Tool `uv`:** Paket manager super cepat dari Astral. Jika belum ada:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

---

## 🛠️ 2. Langkah Demi Langkah Setup Environment dengan `uv`

Buka terminal **PowerShell** dan arahkan ke direktori proyek:

```powershell
# 1. Masuk ke direktori baseline proyek
cd "D:\Kuliah\Semester 7\Tugas Akhir\Proposal Tugas Akhir\EXPERIMENT\Code\baseline_gaur"

# 2. Buat virtual environment (.venv) dengan uv berbasis Python 3.11
uv venv .venv --python 3.11

# 3. Aktivasi virtual environment
# Pada Windows PowerShell:
.venv\Scripts\Activate.ps1
# (Jika muncul batasan eksekusi skrip, jalankan: Set-ExecutionPolicy Unrestricted -Scope Process)
```

---

## 📦 3. Instalasi PyTorch dengan Dukungan CUDA 12.x

Jalankan instalasi PyTorch yang terkompilasi khusus untuk CUDA (Compute 8.6 untuk arsitektur Ampere RTX 3070 Ti):

```powershell
# Install PyTorch + Torchaudio dengan build CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependensi pelengkap dari requirements.txt
uv pip install -r requirements.txt
```

---

## 🔍 4. Verifikasi Status Akselerasi CUDA di Terminal

Jalankan skrip satu baris berikut untuk memverifikasi bahwa PyTorch telah mendeteksi GPU RTX 3070 Ti secara aktif:

```powershell
python -c "import torch; print('CUDA Available   :', torch.cuda.is_available()); print('Device Count     :', torch.cuda.device_count()); print('Device Name      :', torch.cuda.get_device_name(0)); print('CUDA Version     :', torch.version.cuda); print('VRAM Total (GB)  :', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))"
```

**Ekspektasi Output:**
```text
CUDA Available   : True
Device Count     : 1
Device Name      : NVIDIA GeForce RTX 3070 Ti
CUDA Version     : 12.1
VRAM Total (GB)  : 8.0
```

---

## 🚀 5. Cara Menjalankan Eksperimen pada CUDA

### Opsi A: Eksekusi Cepat Single Traversal (*Smoke Test*)
Menguji transisi emosi diagonal dari **Kuadran 3 (Sad)** ke **Kuadran 1 (Happy)** pada dataset 1.802 lagu DEAM:

```powershell
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7
```

**Artefak yang Dihasilkan di `experiments/results/`:**
* `metrics_comparison_deam.csv`: Perbandingan metrik jarak Cosine vs Euclidean vs Manhattan.
* `baseline_cosine_deam_playlist.csv`: Urutan 12 lagu hasil rekomendasi playlist.
* `baseline_deam_traversal_plot.png`: Grafik visualisasi pergerakan emosi 2D pada Russell Circumplex Model.

---

### Opsi B: Batch Benchmark Lintas 6 Rute 4-Kuadran
Mengevaluasi ratusan playlist melintasi rute Diagonal, Horizontal, dan Vertikal menggunakan paralelisasi tensor PyTorch CUDA:

```powershell
python experiments/run_quadrant_experiments.py
```

Skrip ini akan menguji kombinasi:
- **6 Konfigurasi Rute:** $Q_3 \to Q_1$, $Q_2 \to Q_4$, $Q_1 \to Q_2$, $Q_3 \to Q_4$, $Q_1 \to Q_4$, $Q_2 \to Q_3$.
- **Variasi Nilai $K$ Tetangga:** $K \in \{5, 7, 15\}$.
- **Panjang Playlist $N$:** $N \in \{5, 12, 19\}$.
- **Metrik Jarak:** Cosine Similarity vs Euclidean Distance.

**Hasil Output:**
* Tabel komparasi lengkap tersimpan di: `experiments/results/batch_quadrant_experiments_deam.csv`.

---

### Opsi C: Ekstraksi Fitur Akustik Audio DEAM (Batch MP3)
Jika Anda ingin mengekstrak ulang 15 fitur akustik audio mentah dari 1.802 file `.mp3` di `EXPERIMENT/Dataset/DEAM/DEAM_audio/MEMD_audio`:

```powershell
python src/deam_extractor.py
```

---

### Opsi D: Partisi Dataset DEAM (Stratified Split & Anti-Data Leakage)
Membagi 1.802 lagu DEAM secara terstratifikasi berdasarkan 4 Kuadran Emosi Russell (70% Train / 15% Val / 15% Test) untuk memastikan *Zero Data Leakage*:

```powershell
python -c "from src import load_or_generate_deam_dataset, stratified_split_deam, assign_emotion_quadrant, save_split_manifests; df, _ = load_or_generate_deam_dataset(); df['quadrant'] = assign_emotion_quadrant(df); tr, va, te = stratified_split_deam(df, 0.7, 0.15, 0.15); save_split_manifests(tr, va, te, 'experiments/splits'); print('Manifest split sukses disimpan!')"
```

**Artefak yang Dihasilkan di `experiments/splits/`:**
* `deam_train_manifest.csv`: 1.261 lagu (70% data latih / fitting transformer Yeo-Johnson & PCA).
* `deam_val_manifest.csv`: 270 lagu (15% data validasi penyesuaian hyperparameter $K$ dan $N$).
* `deam_test_manifest.csv`: 271 lagu (15% data uji holdout murni bebas data leakage).
* `split_summary.json`: Rekapitulasi jumlah dan persentase trek per kuadran secara presisi.
*Fitur akan tersimpan ke `data/processed/deam_extracted_features.csv`.*

---

## ⚡ 6. Mengapa Menggunakan Akselerasi CUDA pada Baseline Gaur?

| Komponen Eksperimen | Eksekusi CPU Murni | Akselerasi CUDA PyTorch (RTX 3070 Ti) |
| :--- | :--- | :--- |
| **Stage 2 Distance Re-ranking** | Serial per kandidat ($O(K \times D)$) | Paralel serentak pada CUDA cores via `torch.nn.functional.cosine_similarity` |
| **Batch Benchmark 120k Playlist** | Membutuhkan waktu $\approx 3-4$ jam | Selesai dalam $\approx 2-5$ menit melalui tensor batching |
| **Penggunaan VRAM** | 0 MB VRAM (Beban di RAM Sistem) | $\approx 10-50$ MB VRAM (Sangat hemat pada kapasitas 8GB VRAM 3070 Ti) |
| **Kesiapan Model Lanjutan** | Tidak siap untuk deep learning | Siap langsung menampung model usulan Trajectory RNN/Transformer |

---

## 🔧 7. Troubleshooting & Solusi Masalah Umum

### 1. `torch.cuda.is_available()` mengembalikan `False`
* **Penyebab:** PyTorch terinstal versi CPU-only dari indeks default PyPI.
* **Solusi:** Hapus dan instal ulang dengan flag `--index-url`:
  ```powershell
  uv pip uninstall torch torchvision torchaudio
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### 2. Error `ExecutionPolicy` saat aktivasi virtualenv di PowerShell
* **Solusi:** Buka PowerShell dan jalankan:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### 3. Out of Memory (OOM) pada VRAM
* **Solusi:** Model baseline KNN hanya mengonsumsi $< 100$ MB VRAM. Jika OOM terjadi, pastikan tidak ada proses training model besar lain (misal: Stable Diffusion atau LLM) yang sedang berjalan di background. Cek dengan `nvidia-smi`.

---

## 🐙 8. Panduan Git Push ke Repositori GitHub

Karena proyek telah dilengkapi `.gitignore`, seluruh file biner besar (audio MP3, cache tensor `.pt`, dan folder `.venv`) otomatis diabaikan:

```powershell
# 1. Inisialisasi repo (jika belum)
git init
git branch -M main

# 2. Stage seluruh kode sumber dan dokumentasi
git add .

# 3. Commit
git commit -m "feat: complete baseline Gaur KNN on DEAM with PyTorch CUDA RTX 3070 Ti"

# 4. Tambahkan remote repository GitHub Anda
git remote add origin https://github.com/USERNAME/REPO-NAME.git

# 5. Push ke GitHub
git push -u origin main
```
