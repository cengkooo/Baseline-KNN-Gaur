# ⚡ Tutorial: Menjalankan Baseline Eksperimen dengan Akselerasi CUDA (NVIDIA GeForce RTX 3070 Ti)

> **Repositori:** `~/TA-123140205/Code/Baseline-KNN-Gaur/`  
> **Target Hardware:** GPU NVIDIA GeForce RTX 3070 Ti (8GB GDDR6X VRAM, CUDA Compute Capability 8.6)  
> **OS:** CachyOS Linux (Arch-based) — semua perintah memakai shell `bash`  
> **Environment Manager:** `uv` (Astral) + CPython 3.11  
> **Framework:** PyTorch 2.x dengan wheel CUDA 12.8 (`cu128`)  
> **Dataset Utama:** DEAM (1.802 audio tracks & anotasi statis/dinamis)

---

## 📋 1. Prasyarat Sistem & Driver

1. **Driver NVIDIA proprietary/open:** versi $\ge 570$ (mendukung CUDA 12.8+). Di CachyOS:
   ```bash
   sudo pacman -S --needed nvidia-utils nvidia-settings
   ```
2. **Cek driver GPU:**
   ```bash
   nvidia-smi
   ```
   *Pastikan `NVIDIA GeForce RTX 3070 Ti` terdeteksi, beserta versi driver dan `CUDA UMD Version`-nya.*
3. **Python:** `uv` akan mengunduh CPython **3.11** secara otomatis.
   > Jangan memakai `python3` sistem (CachyOS: 3.14) — `librosa`/`numba` belum stabil di sana.
4. **Tool `uv`:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.local/bin/env    # atau buka terminal baru
   uv --version
   ```

---

## 🛠️ 2. Langkah Demi Langkah Setup Environment dengan `uv`

```bash
# 1. Masuk ke direktori proyek
cd ~/TA-123140205/Code/Baseline-KNN-Gaur

# 2. Buat virtual environment (.venv) berbasis Python 3.11
uv venv .venv --python 3.11

# 3. Aktivasi virtual environment
source .venv/bin/activate

# (kalau shell-nya fish)
# source .venv/bin/activate.fish

# Pastikan interpreter yang aktif sudah benar
which python && python -V
```

---

## 📦 3. Instalasi PyTorch dengan Dukungan CUDA 12.8

```bash
# Wheel CUDA besar (±5 GB). Set timeout lebih panjang agar tidak gagal di tengah jalan.
export UV_HTTP_TIMEOUT=300

# Install PyTorch + Torchaudio dengan build CUDA 12.8 (Compute 8.6, Ampere)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependensi pelengkap dari requirements.txt
# (torch>=2.1.0 sudah terpenuhi oleh build CUDA di atas, jadi tidak diunduh ulang)
uv pip install -r requirements.txt
```

---

## 🔍 4. Verifikasi Status Akselerasi CUDA di Terminal

```bash
python -c "import torch; print('CUDA Available   :', torch.cuda.is_available()); print('Device Count     :', torch.cuda.device_count()); print('Device Name      :', torch.cuda.get_device_name(0)); print('CUDA Version     :', torch.version.cuda); print('VRAM Total (GB)  :', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))"
```

**Ekspektasi Output (RTX 3070 Ti Laptop GPU, driver 620.x):**
```text
CUDA Available   : True
Device Count     : 1
Device Name      : NVIDIA GeForce RTX 3070 Ti Laptop GPU
CUDA Version     : 12.8
VRAM Total (GB)  : 7.68
```

---

## 🚀 5. Cara Menjalankan Eksperimen pada CUDA

### Opsi A: Eksekusi Cepat Single Traversal (*Smoke Test*)
Menguji transisi emosi diagonal dari **Kuadran 3 (Sad)** ke **Kuadran 1 (Happy)**:

```bash
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7
```

**Output yang diharapkan** (tanpa dataset DEAM → otomatis sintetis, ±5 detik):
```text
[⚡ GPU ACTIVE] Using: NVIDIA GeForce RTX 3070 Ti Laptop GPU | CUDA: 12.8
[RTX 3070 Ti VRAM] Allocated: 0.00 MB | Reserved: 0.00 MB
[*] Generating synthetic dataset (1,500 tracks)...
[+] PCA 12 Components Fitted. Cumulative Explained Variance: 85.70%
[+] KD-Tree indexed over 1500 unique coordinate points.
[V] DEAM Baseline experiment execution completed successfully!
```

**Artefak yang dihasilkan di `experiments/results/`:**
* `metrics_comparison_deam.csv`: Perbandingan metrik jarak Cosine vs Euclidean vs Manhattan.
* `baseline_cosine_deam_playlist.csv`: Urutan 12 lagu hasil rekomendasi playlist.
* `baseline_deam_traversal_plot.png`: Grafik visualisasi pergerakan emosi 2D pada Russell Circumplex Model.

### Opsi B: Batch Benchmark Lintas 6 Rute 4-Kuadran
Mengevaluasi ratusan playlist melintasi rute Diagonal, Horizontal, dan Vertikal:

```bash
python experiments/run_quadrant_experiments.py
```

Skrip ini akan menguji kombinasi:
- **6 Konfigurasi Rute:** $Q_3 \to Q_1$, $Q_2 \to Q_4$, $Q_1 \to Q_2$, $Q_3 \to Q_4$, $Q_1 \to Q_4$, $Q_2 \to Q_3$.
- **Variasi Nilai $K$ Tetangga:** $K \in \{5, 7, 15\}$.
- **Panjang Playlist $N$:** $N \in \{5, 12, 19\}$.
- **Metrik Jarak:** Cosine Similarity vs Euclidean Distance.

**Hasil Output:** `experiments/results/batch_quadrant_experiments_deam.csv`.

### Opsi C: Ekstraksi Fitur Akustik Audio DEAM (Batch MP3)
Jika ingin mengekstrak ulang 15 fitur akustik dari 1.802 file `.mp3` di `Dataset/DEAM/DEAM_audio/MEMD_audio`:

```bash
python src/deam_extractor.py
```
*Fitur tersimpan ke `data/processed/deam_extracted_features.csv`.*

---

### Opsi D: Partisi Dataset DEAM (Stratified Split & Anti-Data Leakage)
Membagi 1.802 lagu DEAM secara terstratifikasi berdasarkan 4 Kuadran Emosi Russell (70% Train / 15% Val / 15% Test) untuk memastikan *Zero Data Leakage*:

```bash
python -c "from src import load_or_generate_deam_dataset, stratified_split_deam, assign_emotion_quadrant, save_split_manifests; df, _ = load_or_generate_deam_dataset(); df['quadrant'] = assign_emotion_quadrant(df); tr, va, te = stratified_split_deam(df, 0.7, 0.15, 0.15); save_split_manifests(tr, va, te, 'experiments/splits'); print('Manifest split sukses disimpan!')"
```

**Artefak yang Dihasilkan di `experiments/splits/`:**
* `deam_train_manifest.csv`: 1.261 lagu (70% data latih / fitting transformer Yeo-Johnson & PCA).
* `deam_val_manifest.csv`: 270 lagu (15% data validasi penyesuaian hyperparameter $K$ dan $N$).
* `deam_test_manifest.csv`: 271 lagu (15% data uji holdout murni bebas data leakage).
* `split_summary.json`: Rekapitulasi jumlah dan persentase trek per kuadran secara presisi.

---

## 📁 6. Catatan Dataset DEAM

`src/deam_loader.py` mencari anotasi di:

```text
Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level/*.csv
Dataset/DEAM/DEAM_audio/MEMD_audio/{song_id}.mp3
```

Kalau folder `Dataset/DEAM/` tidak ada:
- `run_experiment.py` otomatis memakai **dataset sintetis 1.500 track** (pipeline tetap bisa diuji).
- `run_quadrant_experiments.py` akan gagal karena wajib membaca anotasi DEAM.

Jalankan dengan path lain bila dataset disimpan di luar repo:
```bash
python experiments/run_experiment.py --deam_path /path/ke/Dataset/DEAM
```

---

## ⚡ 7. Seberapa Penting CUDA untuk Baseline Ini?

| Komponen Eksperimen | CPU (NumPy/SciPy) | PyTorch CUDA (RTX 3070 Ti) |
| :--- | :--- | :--- |
| **Stage 2 distance re-ranking** | Loop serial per kandidat | `torch.cdist` / `cosine_similarity` di CUDA cores |
| **Beban komputasi aktual** | $K \le 15$ kandidat × 12 dimensi per langkah | sama, hanya dipindah ke GPU |
| **Waktu batch benchmark** | Detik | Detik (perbedaan tidak terasa) |
| **Penggunaan VRAM** | 0 MB | $\approx 10\text{–}50$ MB |
| **Kesiapan model lanjutan** | Terbatas | Siap menampung Trajectory RNN/Transformer |

**Kesimpulan jujur:** baseline KNN ini komputasinya sangat kecil, jadi CUDA **bukan** untuk
mempercepat baseline — CUDA dipakai agar pipeline sudah siap untuk model deep learning
lanjutan dan agar jalur kode GPU (`src/utils_cuda.py`, `src/stage2_selector.py`) terverifikasi jalan.

---

## 🔧 8. Troubleshooting (Linux / CachyOS)

### 1. `torch.cuda.is_available()` mengembalikan `False`
* **Penyebab:** PyTorch terinstal dari PyPI default, bukan wheel CUDA.
* **Solusi:**
  ```bash
  uv pip uninstall torch torchaudio
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
* **Cek juga:** `nvidia-smi` harus jalan tanpa error dan driver sudah ter-load
  (`lsmod | grep nvidia`). Kalau `nvidia-smi` gagal, driver belum terpasang benar.

### 2. Install torch gagal: `error sending request for url (https://pypi.nvidia.com/...)`
* **Penyebab:** unduhan wheel NVIDIA besar dan koneksi lambat.
* **Solusi:** ulangi dengan timeout lebih panjang (cache `uv` akan melanjutkan file yang sudah ada):
  ```bash
  export UV_HTTP_TIMEOUT=300
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

### 3. `uv venv` memakai Python 3.14 / dependency gagal build
* **Solusi:** buat ulang venv dengan versi eksplisit:
  ```bash
  rm -rf .venv && uv venv .venv --python 3.11
  ```

### 4. Modul tidak ditemukan setelah aktivasi
* **Penyebab:** venv belum aktif, atau perintah dijalankan dengan `python` sistem.
* **Solusi:** `source .venv/bin/activate`, lalu pastikan `which python` menunjuk ke
  `.../Baseline-KNN-Gaur/.venv/bin/python`. Alternatif tanpa aktivasi:
  ```bash
  uv run python experiments/run_experiment.py
  ```

### 5. Out of Memory (OOM) pada VRAM
* Model baseline KNN hanya memakai $< 100$ MB VRAM. Kalau OOM, cek proses lain lewat `nvidia-smi`.

---

## 🐙 9. Panduan Git Push ke Repositori GitHub

`.gitignore` sudah mengecualikan file biner besar (audio MP3, cache tensor `.pt`, `.venv`):

```bash
# 1. Inisialisasi repo (jika belum)
git init
git branch -M main

# 2. Stage seluruh kode sumber dan dokumentasi
git add .

# 3. Commit
git commit -m "feat: complete baseline Gaur KNN on DEAM with PyTorch CUDA RTX 3070 Ti"

# 4. Tambahkan remote repository GitHub
git remote add origin https://github.com/USERNAME/REPO-NAME.git

# 5. Push ke GitHub
git push -u origin main
```
