---
title: "Panduan & Todolist Eksperimen Baseline KNN (Gaur & Donnelly 2023/2024) — DEAM Dataset Utama, PyTorch CUDA RTX 3070 Ti & uv"
tags:
  - todolist
  - baseline
  - knn
  - eksperimen
  - gaur2024
  - deam
  - dataset
  - pytorch
  - cuda
  - rtx3070ti
  - uv
  - github
  - tugas-akhir
status: "READY FOR EXECUTION"
primary_dataset: "DEAM (Database for Emotion Analysis in Music) — 1.802 audio tracks (.mp3)"
dataset_path: "EXPERIMENT/Dataset/DEAM"
hardware_target: "NVIDIA GeForce RTX 3070 Ti (8GB GDDR6X, CUDA Compute 8.6)"
python_manager: "uv (Astral)"
framework: "PyTorch 2.x + CUDA 12.x / 11.8"
repository_target: "GitHub Public/Private Repo (Open Data ITERA Bab 8)"
---

# 🧪 Panduan & Todolist Eksperimen Baseline KNN (Gaur & Donnelly)
## 🎯 Dataset Utama: DEAM (Database for Emotion Analysis in Music)
### ⚡ Akselerasi PyTorch CUDA (NVIDIA RTX 3070 Ti) & Manajemen Environment `uv`

> **Rujukan Teori Utama:** Tesis Shaurya Gaur (*"Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"*, University of Victoria 2023 / Springer EvoMUSART 2024).  
> **Dataset Utama yang Digunakan:** **DEAM (Database for Emotion Analysis in Music)** yang berlokasi lokal di `EXPERIMENT/Dataset/DEAM/`. Dataset ini memuat **1.802 berkas audio klip musik penuh** (`.mp3` durasi 45 detik) beserta anotasi statis per lagu dan anotasi dinamis beresolusi temporal tinggi (sampling 2 Hz per 500 ms).  
> **Target Hardware & Komputasi:** GPU NVIDIA GeForce RTX 3070 Ti (8GB GDDR6X, Compute Capability 8.6), PyTorch 2.x dengan CUDA backend, lingkungan virtual terisolasi menggunakan `uv` (Astral), serta repositori terstandarisasi siap rilis ke GitHub (mematuhi kewajiban *Open Data* Tugas Akhir IF ITERA Bab 8).  
> **Tujuan Dokumen:** Memandu eksekusi bertahap (*step-by-step*) replikasi model acuan (*baseline*) Gaur & Donnelly menggunakan dataset utama DEAM, mengimplementasikan optimasi tensor PyTorch pada GPU RTX 3070 Ti, dan menyiapkan landasan komparasi langsung (*head-to-head benchmark*) terhadap model usulan berbasis lintasan emosi dinamis (*Trajectory Alignment*).

---

## 🎵 1. Profil Dataset Utama: DEAM (`EXPERIMENT/Dataset/DEAM`)

Dataset DEAM tersimpan lengkap secara lokal pada repositori ini di `EXPERIMENT/Dataset/DEAM/` dengan struktur dan karakteristik berikut:

```text
EXPERIMENT/Dataset/DEAM/
├── DEAM_audio/
│   └── MEMD_audio/                                              # 📁 1.802 file audio asli (.mp3, durasi 45 detik, sample rate 44.1 kHz)
│       ├── 2.mp3
│       ├── 3.mp3
│       └── ... (hingga 2058.mp3)
└── DEAM_Annotations/
    └── annotations/
        └── annotations averaged per song/
            ├── song_level/                                      # 🏷️ Anotasi Statis (Digunakan untuk Baseline Gaur)
            │   ├── static_annotations_averaged_songs_1_2000.csv # Kolom: song_id, valence_mean, valence_std, arousal_mean, arousal_std
            │   └── static_annotations_averaged_songs_2000_2058.csv
            └── dynamic (per second annotations)/                # 📈 Anotasi Dinamis 2 Hz (Jembatan ke Model Usulan TA)
                ├── valence.csv                                  # Time-series Valence per 500 ms (sample_15000ms s/d sample_60000ms)
                └── arousal.csv                                  # Time-series Arousal per 500 ms (sample_15000ms s/d sample_60000ms)
```

### 📐 Formulasi Konversi Skala Nilai Emosi DEAM ke Model Gaur
Anotasi emosi statis pada DEAM menggunakan skala Likert / Self-Assessment Manikins (SAM) **[1.0, 9.0]** dengan nilai netral di **5.0**. Sedangkan model Gaur beroperasi pada rentang **$[-1.0, +1.0]$** dengan titik netral di **0.0**.

Rumus transformasi linier baku yang diterapkan:
$$V_{[-1, 1]} = \frac{V_{\text{DEAM}} - 5.0}{4.0}, \quad A_{[-1, 1]} = \frac{A_{\text{DEAM}} - 5.0}{4.0}$$

*Contoh:*
- Nilai $9.0$ (maksimum senang/berenergi) $\to +1.0$
- Nilai $5.0$ (netral) $\to 0.0$
- Nilai $1.0$ (maksimum sedih/tenang) $\to -1.0$

---

## 🖥️ 2. Spesifikasi Lingkungan Komputasi & Standar Repositori

```text
+-----------------------+-------------------------------------------------------------+
| Komponen              | Spesifikasi / Konfigurasi                                   |
+-----------------------+-------------------------------------------------------------+
| Dataset Utama         | DEAM (1.802 audio tracks MP3 + anotasi statis & dinamis)   |
| Kartu Grafis (GPU)    | NVIDIA GeForce RTX 3070 Ti (8GB GDDR6X, Ampere Architecture)|
| Akselerasi Komputasi  | CUDA 12.1 / 12.4 + cuDNN (PyTorch Native Backend)           |
| Audio Feature Lib     | Torchaudio / Librosa (Akselerasi GPU untuk ekstraksi fitur) |
| Python Environment    | uv (Astral) — Virtualenv ultra-cepat & deterministik        |
| Framework Utama       | PyTorch 2.x (torch, torchaudio), scikit-learn, scipy        |
| Kontrol Versi         | Git + GitHub (Clean Commit, .gitignore, Open Data Ready)    |
| Standar Kode Sidang   | PEP 8, Type Hints, Modular, Fully Documented (Walkthrough)  |
+-----------------------+-------------------------------------------------------------+
```

---

## 🗂️ 3. Desain Struktur Folder & File Proyek GitHub-Ready

Struktur direktori di `EXPERIMENT/Code/baseline_gaur/` dirancang modular dan terhubung langsung ke dataset utama `Dataset/DEAM/`:

```text
EXPERIMENT/Code/baseline_gaur/
├── .gitignore                          # Mengabaikan data mentah besar, cache, dan venv
├── pyproject.toml                      # Metadata proyek standar PEP 621 & dependensi uv
├── uv.lock                             # Lockfile dependensi deterministik uv
├── requirements.txt                    # Dependensi standar pip / uv pip
├── README.md                           # Panduan reproduksi eksperimen DEAM & benchmark GPU
├── LICENSE                             # Lisensi open source (MIT License)
├── data/
│   ├── raw/                            # Symlink / referensi path ke ../../../Dataset/DEAM
│   └── processed/                      # Hasil ekstraksi & pembersihan fitur DEAM
│       ├── deam_merged_static.csv      # 1.802 lagu DEAM (song_id, V_norm, A_norm, audio features)
│       ├── deam_pca_features.npy       # Matriks 12 komponen PCA fitur audio DEAM
│       └── features_tensor_gpu.pt      # Tensor PyTorch float32 siap pin ke VRAM 3070 Ti
├── src/
│   ├── __init__.py
│   ├── utils_cuda.py                   # ⚡ Helper deteksi CUDA, VRAM 3070 Ti, & memory management
│   ├── deam_extractor.py               # 🎙️ Ekstraksi 15 fitur audio akustik dari MEMD_audio via GPU/CPU
│   ├── preprocessing.py               # ⚙️ Pipeline PCA 12-D, PowerTransformer, & Normalisasi [-1, 1]
│   ├── dataset.py                     # 📦 Kelas `SongDatasetClass` (KD-Tree CPU + Tensor GPU)
│   ├── stage1_knn.py                  # 🎯 Tahap 1: Interpolasi Waypoint (Persamaan 5) & KNN Retrieval
│   ├── stage2_selector.py             # 🎵 Tahap 2: Audio Distance Re-Ranking via PyTorch Tensor CUDA
│   ├── playlist_generator.py          # 🔄 Loop Dua Tahap (Algoritma 1) terakselerasi GPU
│   └── metrics.py                     # 📊 Metrik Smoothness (PCC) & Evenness (Var(RMSE)/sqrt(D))
├── experiments/
│   ├── configs/
│   │   └── params_deam.json           # Konfigurasi parameter (k: 5–30, N: 5–19, batch_size: 512)
│   ├── run_experiment.py              # Skrip eksekusi cepat single traversal pada DEAM
│   ├── run_quadrant_experiments.py    # 🚀 Batch benchmark 120.000 playlist DEAM via PyTorch CUDA
│   └── results/                       # Hasil output metrik CSV dan plot visualisasi 2D
│       ├── metrics_comparison_deam.csv
│       ├── baseline_cosine_deam_playlist.csv
│       └── baseline_deam_traversal_plot.png
└── notebooks/
    ├── 01_deam_eda_and_features.ipynb # Analisis eksploratif distribusi 1.802 lagu DEAM di kuadran
    └── 02_plot_playlist_paths.ipynb   # Visualisasi trajektori ruang emosi Russell 2D pada DEAM
```

---

## 🚀 4. Checklist Rinci Fase Pengerjaan (Step-by-Step)

### 🔹 FASE 0: Inisialisasi Environment `uv`, PyTorch CUDA, & GitHub Setup

- [ ] **Langkah 0.1 — Setup Virtual Environment Menggunakan `uv`:**
  - Buat dan aktifkan virtual environment berbasis Python 3.11:
    ```powershell
    cd "EXPERIMENT/Code/baseline_gaur"
    uv venv .venv --python 3.11
    .venv\Scripts\activate
    ```

- [ ] **Langkah 0.2 — Instalasi PyTorch CUDA 12.x / 11.8 untuk RTX 3070 Ti:**
  - Pasang paket PyTorch dan pustaka pemrosesan audio/numerik:
    ```powershell
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    uv pip install numpy pandas scipy scikit-learn matplotlib tqdm librosa soundfile
    ```
  - Kunci dependensi dengan `uv pip freeze > requirements.txt`.

- [ ] **Langkah 0.3 — Modul Helper CUDA & Pemantauan VRAM (`src/utils_cuda.py`):**
  - Implementasikan fungsi deteksi device RTX 3070 Ti dan monitoring VRAM:
    ```python
    import torch

    def get_torch_device(prefer_cuda: bool = True) -> torch.device:
        if prefer_cuda and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[⚡ GPU ACTIVE] Device: {device_name} (CUDA {torch.version.cuda})")
            return torch.device('cuda:0')
        print("[ℹ️ CPU ACTIVE] CUDA tidak aktif, menggunakan CPU.")
        return torch.device('cpu')

    def log_vram():
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
            res = torch.cuda.memory_reserved(0) / (1024 ** 2)
            print(f"[RTX 3070 Ti VRAM] Allocated: {alloc:.2f} MB | Reserved: {res:.2f} MB")
    ```

- [ ] **Langkah 0.4 — Konfigurasi Git & Persiapan Repositori GitHub:**
  - Siapkan file `.gitignore` agar file audio besar (4.5 GB) tidak terunggah ke GitHub, melainkan diarahkan melalui dokumentasi *Open Data*:
    ```gitignore
    .venv/
    __pycache__/
    *.pyc
    *.pt
    *.npy
    data/processed/*.csv
    experiments/results/*.csv
    experiments/results/*.png
    ```

---

### 🔹 FASE 1: Ekstraksi Fitur Akustik Audio DEAM & Transformasi Data (`deam_extractor.py`, `preprocessing.py`)

- [ ] **Langkah 1.1 — Parsing Anotasi Emosi Statis DEAM:**
  - Baca file `EXPERIMENT/Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv` dan `songs_2000_2058.csv`.
  - Gabungkan seluruh 1.802 anotasi lagu, pastikan kolom `song_id`, `valence_mean`, dan `arousal_mean` terisi lengkap tanpa missing values.
  - Terapkan rumus normalisasi skala dari $[1.0, 9.0]$ ke $[-1.0, +1.0]$:
    $$V_{\text{norm}} = \frac{\text{valence\_mean} - 5.0}{4.0}, \quad A_{\text{norm}} = \frac{\text{arousal\_mean} - 5.0}{4.0}$$

- [ ] **Langkah 1.2 — Ekstraksi Fitur Audio Akustik dari `EXPERIMENT/Dataset/DEAM/DEAM_audio/MEMD_audio/`:**
  - Buat skrip `src/deam_extractor.py` menggunakan `torchaudio` / `librosa` untuk mengekstrak 15 fitur audio akustik (representasi yang setara dengan fitur audio Spotify & MSD pada metodologi Gaur):
    1. MFCC (koefisien 1 s/d 8) — tekstur timbre vokal/instrumen
    2. Spectral Centroid — kecerahan suara (*brightness*)
    3. Spectral Rolloff — distribusi frekuensi atas
    4. Spectral Flatness — kemiripan suara dengan noise vs nada
    5. Zero Crossing Rate (ZCR) — frekuensi perkusi / desisan
    6. RMS Energy — intensitas kekuatan sinyal audio
    7. Tempo (BPM) — kecepatan ritme lagu
    8. Chroma Energy — distribusi kelas nada harmonik
  - Eksekusi ekstraksi secara batch pada GPU RTX 3070 Ti dengan *multiprocessing dataloader* agar pemrosesan 1.802 lagu selesai dalam hitungan menit.
  - Gabungkan matriks fitur audio dengan koordinat emosi statis ke `data/processed/deam_merged_static.csv`.

- [ ] **Langkah 1.3 — Reduksi Dimensi Fitur Audio (PCA 12-D):**
  - Pisahkan kolom emosi $(V_{\text{norm}}, A_{\text{norm}})$ dari 15 fitur akustik audio. **Koordinat emosi V-A tidak boleh di-PCA**.
  - Terapkan `sklearn.decomposition.PCA(n_components=12)` pada fitur audio akustik.
  - *Validasi:* Pastikan 12 komponen mempertahankan $\ge 90-95\%$ variansi fitur audio DEAM.

- [ ] **Langkah 1.4 — Peredaman Outlier & Normalisasi Skala MinMax:**
  - Terapkan `KBinsDiscretizer(n_bins=20)` pada fitur yang memiliki outlier $> 5.5 \times \text{IQR}$.
  - Terapkan Yeo-Johnson `PowerTransformer` untuk stabilitas distribusi.
  - Normalisasi seluruh fitur audio hasil PCA ke rentang $[-1.0, +1.0]$ menggunakan `MinMaxScaler`.
  - Simpan matriks ke `data/processed/deam_pca_features.npy` dan buat tensor PyTorch GPU-pinned `data/processed/features_tensor_gpu.pt`.

- [x] **Langkah 1.5 — Stratified Train/Val/Test Split Berbasis 4 Kuadran & Anti-Data Leakage (`src/split_utils.py`):**
  - Mengatasi ketidakseimbangan kuadran DEAM (Q3 41.2%, Q1 33.0%, Q4 13.1%, Q2 12.7%) dengan Stratified Sampling (70% Train: 1.261 lagu, 15% Val: 270 lagu, 15% Test: 271 lagu).
  - Terapkan aturan *Zero Data Leakage*: fitting Yeo-Johnson dan PCA hanya pada Train set, lalu panggil `.transform()` ke Val dan Test set.
  - Ekspor manifest partisi ke `experiments/splits/` (`deam_train_manifest.csv`, `deam_val_manifest.csv`, `deam_test_manifest.csv`, `split_summary.json`).

---

### 🔹 FASE 2: Pembangunan Kelas Data & Indeksasi Hibrida (`dataset.py`)

- [ ] **Langkah 2.1 — Implementasi Kelas `SongDatasetClass` untuk DEAM:**
  - Simpan metadata 1.802 lagu DEAM (ID, judul, koordinat $(V_{\text{norm}}, A_{\text{norm}})$) dalam hash table memory CPU.
  - Muat seluruh matriks fitur audio 12-D (1.802 $\times$ 12 tensor float32) langsung ke memori GPU RTX 3070 Ti (hanya membutuhkan memori VRAM sangat kecil: $\approx 0.08$ MB).

- [ ] **Langkah 2.2 — Indeksasi Ruang Emosi DEAM dengan KD-Tree:**
  - Ambil seluruh koordinat $(V_{\text{norm}}, A_{\text{norm}})$ unik dari 1.802 lagu DEAM.
  - Bangun indeks spasial `sklearn.neighbors.NearestNeighbors(algorithm='kd_tree', metric='euclidean')` pada koordinat 2D tersebut.

- [ ] **Langkah 2.3 — Hash Mapping Koordinat ke ID Lagu DEAM:**
  - Petakan pasangan koordinat tuple `(round(V, 4), round(A, 4))` ke daftar `song_id` DEAM yang berada pada titik tersebut.

- [ ] **Langkah 2.4 — Generator Data Sintetis Cadangan:**
  - Tetap pertahankan fungsi `generate_synthetic_dataset` untuk unit test instan saat pipeline dijalankan di mesin CI/CD tanpa file audio DEAM.

---

### 🔹 FASE 3: Algoritma Pipeline Dua Tahap Terakselerasi CUDA (`stage1_knn.py`, `stage2_selector.py`, `playlist_generator.py`)

- [ ] **Langkah 3.1 — Formula Interpolasi Waypoint Linier (Persamaan 5 Gaur 2023):**
  - Pada iterasi ke-$L$ (panjang target $N = 12$ lagu), hitung titik target emosi hipotesis $S_T$:
    $$S_T = S_L + \frac{S_N - S_L}{N - L}$$

- [ ] **Langkah 3.2 — Query Tetangga Terdekat & Filter Perulangan (Stage 1):**
  - Cari titik koordinat emosi terdekat di sekitar $S_T$ via KD-Tree untuk memperoleh kandidat pool.
  - Ambil $K = 7$ lagu kandidat teratas dari hash mapping DEAM.
  - **Filter Keamanan:** Hapus lagu yang sudah terpilih dalam playlist berjalan, dan cegah pemilihan lagu target $S_N$ sebelum langkah terakhir ($N-1$).

- [ ] **Langkah 3.3 — Seleksi Akustik Paralel di GPU RTX 3070 Ti (Stage 2):**
  - Pindahkan vektor audio lagu saat ini ($S_L$) dan matriks kandidat ($K \times 12$) ke GPU PyTorch (`torch.cuda`).
  - Hitung jarak kesamaan menggunakan operasi tensor PyTorch:
    * **Cosine Distance (Metrik Terbaik Gaur):**
      ```python
      # Komputasi paralel di CUDA cores RTX 3070 Ti
      cand_tensors = torch.stack([cand['audio_tensor'] for cand in candidates]).to(device)
      ref_tensor = current_song['audio_tensor'].unsqueeze(0).to(device)
      cosine_sim = torch.nn.functional.cosine_similarity(ref_tensor, cand_tensors, dim=-1)
      cosine_dist = 1.0 - cosine_sim
      best_idx = torch.argmin(cosine_dist).item()
      ```
    * **Euclidean Distance:** `torch.cdist(ref_tensor, cand_tensors, p=2.0)`
    * **Manhattan Distance:** `torch.cdist(ref_tensor, cand_tensors, p=1.0)`
  - Pilih lagu dengan jarak minimum dan tambahkan ke playlist.

---

### 🔹 FASE 4: Metrik Evaluasi Kuantitatif Berbasis Tensor (`metrics.py`)

- [ ] **Langkah 4.1 — Implementasi Metrik Smoothness (Kehalusan):**
  - Hitung **Pearson Correlation Coefficient (PCC)** $\rho$ (Persamaan 6 Gaur 2023) antara Valence dan Arousal:
    $$\rho = \left| \frac{\text{Cov}(V, A)}{\sigma_V \sigma_A} \right|$$
  - Nilai mendekati 1.0 merepresentasikan kelurusan lintasan emosi diagonal yang mulus pada dataset DEAM.

- [ ] **Langkah 4.2 — Implementasi Metrik Evenness (Keteraturan Langkah):**
  - Hitung Root Mean Square Error (RMSE) ukuran langkah transisi dinormalisasi dimensi fitur $D$ (Persamaan 7 Gaur 2023):
    $$\text{RMSE}_k = \frac{\|S_{k+1} - S_k\|_2}{\sqrt{D}}$$
  - Hitung **Evenness** sebagai variansi dari ukuran langkah:
    $$\text{Evenness} = \text{Var}(\text{RMSE}_1, \dots, \text{RMSE}_{N-1})$$
  - Variansi mendekati 0.0 menandakan transisi yang seragam dan stabil tanpa lonjakan perubahan drastis antar-lagu DEAM.

---

### 🔹 FASE 5: Eksperimen Penjelajahan 4-Kuadran pada Lagu DEAM di GPU (`run_quadrant_experiments.py`)

- [ ] **Langkah 5.1 — Partisi 1.802 Lagu DEAM ke 4 Kuadran Russell:**
  - Kelompokkan data DEAM berdasarkan nilai $(V_{\text{norm}}, A_{\text{norm}})$:
    * Kuadran I ($V > 0, A > 0$): Happy, Energetic (misal: klip pop/rock up-beat)
    * Kuadran II ($V < 0, A > 0$): Angry, Tense (misal: klip metal/aggressive)
    * Kuadran III ($V < 0, A < 0$): Sad, Depressed (misal: klip melankolis minor)
    * Kuadran IV ($V > 0, A < 0$): Calm, Relaxed (misal: klip akustik tenang/ambient)

- [ ] **Langkah 5.2 — Batch Generation 120.000 Playlist DEAM via GPU Batching:**
  - Jalankan pengujian playlist pada 6 konfigurasi rute: Horizontal (I $\leftrightarrow$ II, III $\leftrightarrow$ IV), Vertikal (I $\leftrightarrow$ IV, II $\leftrightarrow$ III), dan Diagonal (I $\leftrightarrow$ III, II $\leftrightarrow$ IV).
  - Manfaatkan paralelisasi PyTorch CUDA di RTX 3070 Ti untuk mengevaluasi variasi $K \in \{5, 10, 15, 20, 25, 30\}$ dan panjang playlist $N \in \{5, 11, 15, 19\}$ secara efisien dalam hitungan menit.
  - Catat log waktu eksekusi CPU vs GPU RTX 3070 Ti (*speedup benchmark*).
  - Ekspor hasil metrik ke `experiments/results/metrics_comparison_deam.csv` dan simpan visualisasi ke `experiments/results/baseline_deam_traversal_plot.png`.

---

## 🐙 5. Alur Kerja GitHub & Instruksi Reproduksibilitas

```bash
# 1. Clone repositori dari GitHub
git clone https://github.com/username/baseline-mood-dynamic-playlist.git
cd baseline-mood-dynamic-playlist

# 2. Inisialisasi virtual environment menggunakan uv
uv venv .venv --python 3.11
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell

# 3. Sinkronisasi dependensi PyTorch CUDA
uv pip install -r requirements.txt

# 4. Verifikasi status GPU NVIDIA RTX 3070 Ti & DEAM Dataset
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 5. Jalankan ekstraksi fitur DEAM (jika belum diekstrak)
python src/deam_extractor.py --deam_dir "../../Dataset/DEAM"

# 6. Jalankan eksekusi eksperimen baseline DEAM
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7 --use_deam
```

---

## 💡 6. Jembatan Emas: Dari Baseline Gaur DEAM ke Model Usulan Anda

Karena dataset utama yang digunakan adalah **DEAM**, proyek ini memiliki keuntungan ilmiah yang sangat strategis:

| Fitur / Parameter | Baseline Gaur pada DEAM | Model Usulan Tugas Akhir Anda pada DEAM |
| :--- | :--- | :--- |
| **Sumber Data Emosi** | Anotasi Statis (`song_level/*.csv`) | **Anotasi Dinamis 2 Hz (`dynamic/*.csv`)** |
| **Representasi Emosi** | Titik statis tunggal $(\bar{V}_{\text{norm}}, \bar{A}_{\text{norm}})$ | **Vektor Kemiringan Lintasan $\mathbf{v}_{\text{song}} = (\beta_V, \beta_A)$** |
| **Resolusi Temporal** | 1 nilai mewakili 45 detik lagu | **90 titik sampel emosi (per 500 ms) per lagu** |
| **Stage 1 (Retrieval)** | KD-Tree KNN pada titik emosi statis | KD-Tree KNN pada titik awal lintasan lagu |
| **Stage 2 (Selection)** | Jarak akustik audio murni (Cosine) | **Hybrid Scoring: Audio Distance + Sudut Kelurusan Lintasan ($\cos \theta_{\text{traj}}$)** |
| **Masalah Diatasi** | Mengabaikan pergeseran emosi di dalam lagu | **Mengeliminasi diskontinuitas emosi sambungan (*boundary jump*)** |

---

## 🔗 Navigasi & Berkas Terkait
- Dataset Utama: [EXPERIMENT/Dataset/DEAM/](file:///D:/Kuliah/Semester%207/Tugas%20Akhir/Proposal%20Tugas%20Akhir/EXPERIMENT/Dataset/DEAM)
- Kembali ke Indeks: [[00-Index-Todolist]]
- Terkait Metodologi Usulan: [[Phase-03-Bab-3-Metodologi-Desain]]
- Terkait Eksperimen DEAM: [[Phase-04-Coding-Eksperimen-DEAM]]
- Review Paper Gaur: [[Gaur2023-Mood-Dynamic]] | [[Gaur2020-Mood-Arc]]
- Implementasi Kode: `EXPERIMENT/Code/baseline_gaur/`
