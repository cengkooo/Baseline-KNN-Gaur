# Traine 1 — Eksekusi Pipeline Baseline Gaur dengan Fitur Akustik Nyata (Mode B)

> **Tanggal:** 19 November 2026
> **Lokasi:** `~/TA-123140205/Code/Baseline-KNN-Gaur`
> **Hardware:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (8.192 MiB), driver 620.70.06, CUDA 12.8
> **Environment:** CPython 3.11.18, PyTorch 2.11.0+cu128, `torch.cuda.is_available() == True`, uv 0.13.9
> **Dataset:** DEAM, 1.802 MP3 (`Dataset/DEAM/DEAM_audio/MEMD_audio`) + anotasi statis/dinamis
> **Commit akhir:** `3515e16` (kode + hasil), `e7112c1` (dokumen). Push ke GitHub belum (lihat bagian 8).

---

## 1. Ringkasan Satu Paragraf

Sebelum sesi ini, seluruh angka baseline masih memakai **Mode A** — fitur akustik *sintetis deterministik* (`random_state=42`) karena `data/processed/deam_extracted_features.csv` belum ada. Sesi ini mengeksekusi **Fase 5**: mengekstrak 15 fitur akustik nyata (MFCC, spectral, RMS, ZCR, tempo, chroma) dari **1.802 file MP3** memakai librosa, memvalidasi bahwa tidak ada fallback acak yang senyap, lalu **meregenerasi seluruh eksperimen** (single run dan batch 540 playlist) dengan **Mode B**. Di tengah proses ditemukan dan diperbaiki satu bug loader yang membuat 8 kolom anotasi emosi ikut masuk sebagai "fitur akustik". Hasil akhir: PCA 12D dengan explained variance **99.10%**, single run PCC **0.995 / 0.993 / 0.984**, batch **540 baris tanpa NaN**, dan seluruh dokumen sudah disinkronkan.

---

## 2. Kondisi Awal (Sebelum Sesi Ini)

| Komponen | Status awal |
| :--- | :--- |
| `data/processed/deam_extracted_features.csv` | Tidak ada → loader jatuh ke fitur sintetis |
| Single run | Ada, tapi Mode A: PCC Cosine 0.978100 / Euclidean 0.987591 / Manhattan 0.976171 |
| PCA explained variance | 82.26% (Mode A) |
| Batch 540 | Ada, tapi dihasilkan dari fitur sintetis |
| GPU / venv / dataset | Sudah beres (Fase 0–4, 6–8 todolist) |

Artinya: pada kondisi itu, angka Smoothness/Evenness **valid untuk menguji kebenaran algoritma**, tetapi **bukan bukti** bahwa pipeline bekerja pada karakteristik audio nyata. Itu yang diperbaiki di sesi ini.

---

## 3. Yang Dikerjakan

### 3.1 Smoke test ekstraksi (Fase 5.1–5.2)

```bash
python src/deam_extractor.py --limit 5 --output /tmp/opencode/deam_features_smoke.csv
```

Hasil: 5 baris, 16 kolom (15 fitur + `song_id`), 0 NaN, durasi ~8,6 detik (≈1,16 detik/lagu).

| Kolom | Rentang pada 5 lagu | Interpretasi |
| :--- | :--- | :--- |
| `tempo_bpm` | 68,0 – 123,0 | wajar untuk musik pop/rock |
| `spectral_centroid` | 2.125 – 3.252 Hz | bukan nilai acak seragam |
| `rms_energy` | 0,078 – 0,287 | dalam rentang 0–1 |
| `mfcc_1` | -98,2 – -25,7 | tipikal MFCC librosa |

Kesimpulan: ekstraksi benar-benar membaca audio, **bukan** fallback `np.random` di dalam `except` (fallback akan menghasilkan `mfcc_*` ~ N(0,1) dan `spectral_centroid` uniform 1000–4000).

### 3.2 Ekstraksi penuh 1.802 lagu

Ekstraksi sequential diperkirakan ±35 menit (1,16 detik/lagu). Karena mesin punya 20 core dan `extract_features_from_audio()` bersifat murni per-file, ekstraksi dijalankan paralel memakai fungsi yang **sama persis** dari `src/deam_extractor.py`:

- 12 worker `multiprocessing.Pool`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` per worker (menghindari oversubscription)
- Output tetap `data/processed/deam_extracted_features.csv`
- Waktu: **± 4 menit** untuk 1.802 lagu

> Catatan operasional: percobaan pertama memakai `nohup ... &` dan proses berhenti di 500/1.802 karena process group dimatikan saat shell sesi berakhir. Eksekusi ulang di foreground berhasil penuh. Untuk pemakaian biasa (terminal sendiri), `nohup` tetap aman.

### 3.3 Validasi CSV final (Fase 5.5)

```
Shape            : (1802, 16)
song_id unik     : 1802 | duplikat: 0
NaN total        : 0
```

Distribusi final seluruh korpus:

| Kolom | Mean | Median | Min – Max |
| :--- | ---: | ---: | ---: |
| `mfcc_1` | -183,6 | -172,4 | -510,2 – 47,5 |
| `spectral_centroid` | 1.780,2 | 1.761,6 | 249,3 – 4.499,5 |
| `tempo_bpm` | 118,8 | 117,5 | 24,8 – 198,8 |
| `rms_energy` | 0,113 | 0,095 | 0,009 – 0,559 |

Tidak ada indikasi fallback acak: sebaran kontinu, korelasi antar-fitur wajar, dan tidak ada baris yang menyerupai distribusi uniform fallback.

### 3.4 Bug yang ditemukan dan diperbaiki

Setelah CSV nyata ada, loader mengembalikan **23** kolom "fitur akustik", bukan 15:

```
['valence_ max_mean', 'valence_max_std', 'valence_min_mean', 'valence_min_std',
 'arousal_max_mean', 'arousal_max_std', 'arousal_min_mean', 'arousal_min_std',
 'mfcc_1', ..., 'chroma_3']
```

**Penyebab:** `load_or_generate_deam_dataset()` menghitung `audio_cols` sebagai *semua kolom hasil merge yang bukan `meta_cols`*. File anotasi statis kedua (`static_annotations_averaged_songs_2000_2058.csv`) punya 8 kolom tambahan (`*_max_*`, `*_min_*`) yang tidak ada di `meta_cols`. Akibatnya:

- 8 kolom anotasi emosi (informasi target!) bocor ke input preprocessing sebagai "fitur akustik";
- kolom tersebut hanya terisi untuk 58 lagu terakhir, sisanya **NaN**;
- PCA akan di-fit pada matriks yang tercemar dan ber-NaN.

**Perbaikan** (`src/deam_loader.py`): turunkan `audio_cols` langsung dari CSV fitur, bukan dari sisa kolom.

```python
if features_csv_path and os.path.exists(features_csv_path):
    df_feat = pd.read_csv(features_csv_path)
    df_feat.columns = [c.strip() for c in df_feat.columns]
    merged = pd.merge(df_static, df_feat, on='song_id', how='inner')
    # The features CSV contains exactly the acoustic feature columns plus song_id.
    audio_cols = [c for c in df_feat.columns if c != 'song_id']
    return merged, audio_cols
```

Hasil setelah perbaikan: `Fitur dimuat: 15`, 1.802 baris, 0 NaN. Test suite tetap 8/8 `OK`.

### 3.5 Verifikasi ulang pipeline (Fase 3, 6, 8)

| Gate | Hasil |
| :--- | :--- |
| `python tests/test_baseline.py` | `Ran 8 tests in 1.032s — OK` |
| Preprocessing | 1.802 baris, 12 komponen PCA, **explained variance 99.10%**, rentang -1,0 … 1,0, 0 NaN |
| Loader auto-detect | Memakai CSV nyata (`Audio Features: 15`), bukan sintetis |
| KD-Tree | 1.113 titik koordinat unik |

### 3.6 Single run (Fase 8)

```bash
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7
```

Log kunci:

```
[⚡ GPU ACTIVE] Using: NVIDIA GeForce RTX 3070 Ti Laptop GPU | CUDA: 12.8
[+] Active Dataset: DEAM (1,802 Tracks) | Total Tracks: 1802 | Audio Features: 15
[+] PCA 12 Components Fitted. Cumulative Explained Variance: 99.10%
[+] KD-Tree indexed over 1113 unique coordinate points.
Origin (Q3 Sad)  : 'DEAM Clip 2022' [V=-0.300, A=-0.150]
Target (Q1 Happy): 'DEAM Clip 1879' [V=0.100, A=0.350]
```

Metrik (N=12, K=7, Mode B):

| Metrik | Smoothness (PCC) | Evenness V-A Var | Evenness Audio Var | Mean V-A Step | Mean Audio Step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Cosine | **0.995043** | 0.000245 | 0.004586 | 0.042695 | 0.206768 |
| Euclidean | **0.992504** | 0.000288 | 0.002053 | 0.042230 | 0.191666 |
| Manhattan | **0.983973** | 0.000312 | 0.004336 | 0.043303 | 0.201044 |

Playlist Cosine (12 lagu unik, Q3 → Q1):

```
DEAM_2022 (V=-0.300, A=-0.150)
DEAM_0153 (V=-0.275, A=-0.125)
DEAM_0010 (V=-0.250, A=-0.075)
DEAM_0667 (V=-0.200, A=-0.025)
DEAM_1476 (V=-0.125, A= 0.025)
DEAM_1999 (V=-0.100, A= 0.100)
DEAM_1985 (V=-0.050, A= 0.150)
DEAM_1235 (V=-0.025, A= 0.200)
DEAM_1991 (V= 0.000, A= 0.250)
DEAM_1317 (V= 0.075, A= 0.300)
DEAM_1366 (V= 0.100, A= 0.325)
DEAM_1879 (V= 0.100, A= 0.350)
```

Lintasan monoton naik di V dan A — tidak ada zigzag.

### 3.7 Batch 540 playlist (Fase 9)

```bash
python experiments/run_quadrant_experiments.py
```

Konfigurasi yang benar-benar dijalankan: 6 rute × K ∈ {5, 7, 15} × N ∈ {5, 12, 19} × {cosine, euclidean} × 5 trial = **540**. Waktu eksekusi 10,6 detik.

Validasi:

```
Shape      : (540, 12)
Rute       : 6 (masing-masing 90 baris)
K / N      : {5,7,15} / {5,12,19}
NaN        : tidak ada
Smoothness : semua di [-1, 1]; Evenness ≥ 0
```

Agregat per rute (mean):

| Rute | Cosine | Euclidean |
| :--- | ---: | ---: |
| Diagonal Q3→Q1 | 0.990413 | **0.990542** |
| Diagonal Q2→Q4 | **0.967837** | 0.962645 |
| Vertical Q2→Q3 | 0.949509 | **0.950079** |
| Horizontal Q3→Q4 | **0.820197** | 0.816798 |
| Vertical Q1→Q4 | **0.718927** | 0.649495 |
| Horizontal Q1→Q2 | 0.662838 | **0.689925** |

Efek K dan N (mean Smoothness):

| K \ N | 5 | 12 | 19 |
| ---: | ---: | ---: | ---: |
| 5 | 0.9248 | 0.8776 | 0.8159 |
| 7 | 0.9073 | 0.8558 | 0.7891 |
| 15 | 0.8865 | 0.8085 | 0.7615 |

Temuan:

1. Rute **diagonal** paling halus (0.963–0.991); rute yang menyentuh Q2/Q4 (kuadran minoritas, ±13%) paling sulit (0.649–0.820).
2. **N kecil + K kecil** paling halus (0.9248); **N besar + K besar** paling kasar (0.7615).
3. Cosine menang di 3 rute, Euclidean di 3 rute lain; rata-rata Cosine **0.8516** vs Euclidean **0.8432**. Cosine unggul tipis pada kelurusan emosi, Euclidean unggul pada keseragaman transisi akustik (Evenness Audio Var 0.0035 vs 0.0042).
4. Kasus terburuk: Horizontal Q1→Q2, Cosine, K=5, N=19, trial 1 → PCC 0.0042.

---

## 4. Perbandingan Mode A vs Mode B (bahan naskah)

| Aspek | Mode A (sintetis) | Mode B (nyata) |
| :--- | :---: | :---: |
| Sumber fitur | `np.random` deterministik, dikorelasikan ke V/A | librosa dari 1.802 MP3 |
| PCA explained variance | 82.26% | **99.10%** |
| Smoothness Cosine / Euclidean / Manhattan | 0.978100 / 0.987591 / 0.976171 | **0.995043 / 0.992504 / 0.983973** |
| Evenness Audio Var | 0.004930 – 0.005409 | **0.002053 – 0.004586** |
| Mean Audio Step | 0.2895 – 0.3506 | **0.1917 – 0.2068** |

Interpretasi singkat: fitur nyata lebih informatif (variance lebih tinggi, transisi akustik lebih seragam/Evenness lebih kecil), sementara kelurusan emosi tetap tinggi (PCC > 0,98). Ini masuk akal: MFCC/spectral/tempo dari lagu nyata membentuk manifold yang lebih rapat daripada fitur acak.

Angka Mode A **tetap disimpan** di arsip dan di dokumen sebagai pembanding — bukan dihapus — supaya klaim "Mode B lebih baik" bisa diverifikasi, bukan sekadar dinyatakan.

---

## 5. Artefak yang Dihasilkan

| Artefak | Isi |
| :--- | :--- |
| `data/processed/deam_extracted_features.csv` | 1.802 × 16, fitur akustik nyata (gitignored) |
| `experiments/results/metrics_comparison_deam.csv` | Metrik single run 3 metrik (Mode B) |
| `experiments/results/baseline_cosine_deam_playlist.csv` | Playlist 12 lagu Q3→Q1 |
| `experiments/results/baseline_deam_traversal_plot.png` | Plot lintasan pada Russell Circumplex |
| `experiments/results/batch_quadrant_experiments_deam.csv` | 540 baris hasil batch |
| `experiments/results/batch_quadrant_summary_deam.csv` | Agregat 12 baris (rute × metrik) |
| `experiments/results/run_single_deam.log` | Log single run |
| `experiments/results/run_batch_deam.log` | Log batch |
| `experiments/results/archive_20261119_1703/` | Arsip semua artefak di atas |

---

## 6. Dokumentasi yang Disinkronkan

| File | Perubahan |
| :--- | :--- |
| `todolist.md` | Fase 5, 9, 10, 12 ditutup; tabel status, metrik, analisis K/N, dan provenance diperbarui ke Mode B |
| `PANDUAN_VALIDASI_DAN_TESTING.md` | Tabel Metode 2, langkah playlist, kriteria metrik, dan perbandingan Cosine/Euclidean/Manhattan diperbarui |
| `TUTORIAL_RUN_CUDA.md` | Contoh output Mode B, penjelasan auto-detect Mode A/B, cara regenerasi fitur |
| `experiments/configs/params_deam.json` | Disamakan dengan konfigurasi yang benar-benar dijalankan (batch 2 metrik, 540 evaluasi) |
| `.gitignore` | `Dataset`, `data/processed/`, `experiments/results/` seluruhnya di-ignore |

---

## 7. Rekaman Git

```
e7112c1 docs: close phases 5,9,10,12 in todolist; note push pending
3515e16 feat: real acoustic features (Mode B) + regenerated single/batch DEAM results
8a622ff docs: sync todolist with verified phases and single-run metrics
6961158 docs: record successful DEAM single run after conflict fix
```

Diff fungsional hanya dua hal: perbaikan `audio_cols` di `src/deam_loader.py` dan perluasan `.gitignore`. Sisanya data hasil + dokumen.

---

## 8. Yang Belum Selesai / Catatan

1. **Push ke GitHub belum berhasil.** `git push origin main` menggantung lalu timeout dari lingkungan ini (tidak ada kredensial/jaringan). Commit lokal sudah siap; jalankan manual:
   ```bash
   cd ~/TA-123140205/Code/Baseline-KNN-Gaur && git push origin main
   ```
2. **`deam_extracted_features.csv` tidak di-commit** (1.802 baris, ~508 KB, masuk `.gitignore` sebagai artefak data). Di mesin lain, regenerate dulu sebelum menjalankan eksperimen:
   ```bash
   python src/deam_extractor.py        # ± 35 menit sequential, tanpa resume
   ```
3. **Script ekstraksi paralel belum masuk repo.** Ada di `/tmp/opencode/extract_parallel.py` (12 worker, ~4 menit). Bisa dipindah ke `src/` bila ingin dipakai rutin — perlu keputusan, belum dilakukan.
4. **Manhattan tidak ikut batch.** `run_quadrant_experiments.py` hanya menjalankan Cosine dan Euclidean; Manhattan hanya ada di single run. Bila naskah butuh Manhattan di batch, skripnya harus diubah (dan batch 540 akan menjadi 810).
5. **Stash lama masih ada** (`stash@{0}: WIP on main: 50ddf91`) dan sengaja belum di-drop sampai push sukses.
6. **Split 70/15/15 tidak dipakai baseline.** Retrieval baseline memakai seluruh katalog 1.802 lagu (sesuai perilaku asli Gaur). Ini harus dinyatakan eksplisit di Bab Metodologi agar tidak dianggap data leakage.

---

## 9. Cara Reproduksi Cepat

```bash
cd ~/TA-123140205/Code/Baseline-KNN-Gaur
source .venv/bin/activate

# 1. Fitur akustik nyata (lewati jika CSV sudah ada)
python src/deam_extractor.py            # ± 35 menit; atau runner paralel ± 4 menit

# 2. Gate kualitas
python tests/test_baseline.py

# 3. Single run
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7

# 4. Batch 540
python experiments/run_quadrant_experiments.py

# 5. Validasi cepat
python - <<'PY'
import pandas as pd
d = pd.read_csv('experiments/results/batch_quadrant_experiments_deam.csv')
assert len(d) == 540 and d.isna().sum().sum() == 0
print("BATCH OK", d.shape)
PY
```

Ekspektasi akhir: 8/8 test OK, PCA variance 99.10%, single run PCC ≥ 0.98 untuk ketiga metrik, batch 540 baris tanpa NaN.
