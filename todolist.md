# 🚀 Todolist Eksekusi Training/Eksperimen Baseline Gaur (DEAM) — CUDA Lokal RTX 3070 Ti

> **Tujuan dokumen:** daftar langkah lengkap, berurutan, dan dapat dicentang untuk menjalankan seluruh pipeline eksperimen baseline Gaur (2023) di laptop ini dengan akselerasi CUDA, dari kondisi *cold start* sampai artefak siap dipakai di Bab Hasil Tugas Akhir.
>
> **Lokasi proyek:** `~/TA-123140205/Code/Baseline-KNN-Gaur/`
> **Repo Git:** `https://github.com/cengkooo/Baseline-KNN-Gaur.git` (branch `main`)
> **Dataset:** `Dataset/DEAM` (symlink ke `~/TA-123140205/Dataset/DEAM`, 1.4 GB, 1.802 MP3)
> **Hardware terverifikasi:** NVIDIA GeForce RTX 3070 Ti Laptop GPU, driver `620.70.06`, 8.192 MiB VRAM
> **Environment terverifikasi:** `uv 0.13.9`, CPython `3.11.18`, PyTorch `2.11.0+cu128`, `torch.cuda.is_available() == True`, CUDA runtime `12.8`
> **Status terakhir dicek:** 19 November 2026

---

## ⚠️ Catatan penting sebelum mulai

1. **Baseline ini bukan "training" model neural network.** Tidak ada bobot model yang dilatih. Yang dimaksud "training/run" di sini adalah:
   - *fit* pipeline preprocessing (IQR → Yeo-Johnson → PCA 12D → MinMax) pada data DEAM,
   - membangun indeks KD-Tree,
   - menjalankan retrieval + re-ranking untuk menghasilkan playlist,
   - menghitung metrik evaluasi.
   Jadi tidak ada epoch, checkpoint, atau loss curve. GPU dipakai di Stage 2 (jarak akustik) dan sebagai kesiapan jalur CUDA.
2. **Merge conflict sudah diselesaikan (19 Nov 2026).** Tiga file (`src/deam_loader.py`, `README.md`, `TUTORIAL_RUN_CUDA.md`) sudah bersih dan di-commit (`14aa2ad`). Unit test 8/8 `OK`. Fase 0 di bawah disimpan sebagai rekam jejak dan untuk berjaga-jaga bila konflik muncul lagi.
3. **Artefak hasil yang ada sekarang berasal dari dataset sintetis, bukan DEAM.** Bukti: `experiments/results/baseline_cosine_deam_playlist.csv` berisi `track_id` `TRK_000070`, `Song Title 70`, sedangkan DEAM asli ber-ID `DEAM_xxxx`. Artinya run lama jatuh ke fallback sintetis. Artefak ini **harus diregenerasi** setelah Fase 0 selesai, dan wajib diverifikasi ber-ID `DEAM_`.
4. **Gunakan venv proyek, jangan `python` sistem.** Semua perintah di bawah dijalankan dari root proyek dengan `.venv` aktif (atau prefiks `.venv/bin/python`).
5. **Jangan pernah commit** `.venv/`, `Dataset/`, `data/processed/`, dan `experiments/results/*` (sudah di-`.gitignore`).

---

## ⚡ Jalur cepat (ringkasan 8 perintah)

Hanya jalankan ini **setelah Fase 0 selesai**:

```bash
cd ~/TA-123140205/Code/Baseline-KNN-Gaur
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
python tests/test_baseline.py
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7 2>&1 | tee experiments/results/run_single_deam.log
python experiments/run_quadrant_experiments.py 2>&1 | tee experiments/results/run_batch_deam.log
python -c "import pandas as pd; d=pd.read_csv('experiments/results/batch_quadrant_experiments_deam.csv'); print(d.shape); print(d.groupby(['Route','Distance_Metric'])['Smoothness_PCC'].mean())"
git add -A && git commit -m "feat: verified DEAM baseline run on local CUDA RTX 3070 Ti"
```

---

## 📊 Status awal (hasil pemeriksaan langsung di laptop ini)

| Komponen | Status | Bukti / Catatan |
| :--- | :---: | :--- |
| Driver + GPU | ✅ | `nvidia-smi` → RTX 3070 Ti Laptop GPU, 8192 MiB |
| `uv` | ✅ | `/usr/bin/uv` versi 0.13.9 |
| `.venv` Python 3.11 | ✅ | `Python 3.11.18` |
| PyTorch CUDA | ✅ | `2.11.0+cu128`, `cuda.is_available() == True` |
| Dataset DEAM | ✅ | symlink `Dataset` → `~/TA-123140205/Dataset`, 1.802 MP3, anotasi lengkap |
| Manifest split | ✅ | train 1.261 / val 270 / test 271 |
| **Merge conflict** | ✅ Selesai | Commit `14aa2ad`; unit test 8/8 `OK` |
| `data/processed/` | ⚠️ Belum ada | Ekstraksi fitur audio nyata belum pernah dijalankan |
| Hasil batch 540 | ⚠️ Belum ada | `batch_quadrant_experiments_deam.csv` tidak ada |
| Hasil single run | ⚠️ Stale/sintetis | Ada, tetapi `TRK_*` (bukan DEAM) |
| Git working tree | ✅ Bersih | Hanya `?? Dataset` (symlink, memang tidak di-commit) |

---

# 🔹 FASE 0 — Buka Blocker: Selesaikan Merge Conflict (± 10 menit)

> **Gerbang wajib.** Semua fase lain akan gagal selama langkah ini belum tuntas.

- [x] **0.1 — Cek status dan daftar file konflik**
  ```bash
  cd ~/TA-123140205/Code/Baseline-KNN-Gaur
  git status --short
  git diff --name-only --diff-filter=U
  grep -rn '^<<<<<<< \|^=======$\|^>>>>>>> ' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ .
  ```
  Ekspektasi saat ini: `README.md`, `TUTORIAL_RUN_CUDA.md`, `src/deam_loader.py`.

- [x] **0.2 — Perbaiki `src/deam_loader.py` (WAJIB, ini yang memblokir eksekusi)**
  Hapus semua marker `<<<<<<<`, `=======`, `>>>>>>>`, lalu pertahankan versi path yang **relatif ke root repo** karena dataset di laptop ini ada di `Dataset/DEAM` (symlink), bukan `EXPERIMENT/Dataset/DEAM`.
  Ganti seluruh fungsi `get_default_deam_path()` dengan versi bersih berikut:
  ```python
  def get_default_deam_path() -> str:
      """Finds the absolute path to Dataset/DEAM (repo-relative, symlink-safe)."""
      current_file_dir = os.path.dirname(os.path.abspath(__file__))
      repo_root = os.path.abspath(os.path.join(current_file_dir, ".."))
      candidates = [
          os.path.join(repo_root, "Dataset", "DEAM"),   # ~/TA-123140205/Code/Baseline-KNN-Gaur/Dataset/DEAM
          os.path.abspath(os.path.join("Dataset", "DEAM")),
      ]
      for candidate in candidates:
          if os.path.exists(candidate):
              return candidate
      return candidates[0]
  ```
  Jangan mempertahankan fallback Windows `D:\Kuliah\...` — tidak relevan di laptop ini.

- [x] **0.3 — Perbaiki `README.md` dan `TUTORIAL_RUN_CUDA.md`**
  Keduanya hanya konflik dokumentasi. **Gabungkan kedua sisi**, jangan pilih salah satu:
  - `README.md`: pertahankan bagian *DEAM Dataset Splitting Methodology (Anti-Data Leakage)* **dan** bagian *Setup & Execution Guide (CachyOS / Linux)*.
  - `TUTORIAL_RUN_CUDA.md`: pertahankan path `Dataset/DEAM/DEAM_audio/MEMD_audio` (Linux) **dan** Opsi D (stratified split); buang referensi `EXPERIMENT/Dataset` yang tidak ada di laptop ini.

- [x] **0.4 — Verifikasi tidak ada marker tersisa**
  ```bash
  grep -rn '^<<<<<<< \|^=======$\|^>>>>>>> ' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ . || echo "BERSIH"
  python -m py_compile src/*.py experiments/*.py tests/test_baseline.py && echo "SEMUA FILE PYTHON VALID"
  ```

- [x] **0.5 — Tandai konflik selesai dan commit**
  ```bash
  git add src/deam_loader.py README.md TUTORIAL_RUN_CUDA.md src/metrics.py
  git status --short          # pastikan tidak ada lagi kode UU
  git commit -m "fix: resolve merge conflicts and make DEAM path repo-relative"
  git stash list              # stash@{0} biarkan dulu sampai Fase 11
  ```

**Definisi selesai Fase 0:** tidak ada marker konflik, `py_compile` lolos, dan `git status` bersih dari status `UU`.

---

# 🔹 FASE 1 — Preflight Hardware & Dataset (± 2 menit)

- [ ] **1.1 — Pastikan GPU dan driver terdeteksi**
  ```bash
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  ```
  Ekspektasi: `NVIDIA GeForce RTX 3070 Ti Laptop GPU, 620.70.06, 8192 MiB`.

- [ ] **1.2 — Pastikan symlink dataset benar**
  ```bash
  ls -l Dataset
  ls Dataset/DEAM
  find Dataset/DEAM/DEAM_audio/MEMD_audio -maxdepth 1 -name '*.mp3' | wc -l
  ```
  Ekspektasi: symlink ke `/home/itera-if/TA-123140205/Dataset`, dan jumlah MP3 = **1802**.

- [ ] **1.3 — Pastikan file anotasi statis & dinamis lengkap**
  ```bash
  ls "Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level"
  ls "Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)"
  ```
  Ekspektasi statis: `static_annotations_averaged_songs_1_2000.csv`, `static_annotations_averaged_songs_2000_2058.csv`.
  Ekspektasi dinamis: `valence.csv`, `arousal.csv` (dipakai model usulan, bukan baseline).

- [ ] **1.4 — Cek resource laptop**
  ```bash
  df -h ~            # butuh ruang untuk log + CSV (kecil) dan sementara ekstraksi audio
  free -h            # RAM; ekstraksi librosa single-thread ~0.5-1.5 GB
  ```

**Definisi selesai Fase 1:** GPU terdeteksi, 1.802 MP3, 2 CSV statis, 2 CSV dinamis, ruang disk cukup.

---

# 🔹 FASE 2 — Environment `uv` + PyTorch CUDA (± 5 menit, sebagian sudah terpasang)

- [ ] **2.1 — Aktifkan venv dan cek interpreter**
  ```bash
  cd ~/TA-123140205/Code/Baseline-KNN-Gaur
  source .venv/bin/activate
  which python && python -V
  ```
  Ekspektasi: `.../Baseline-KNN-Gaur/.venv/bin/python`, `Python 3.11.18`.

- [ ] **2.2 — Verifikasi CUDA dari dalam PyTorch**
  ```bash
  python -c "import torch; print('CUDA Available   :', torch.cuda.is_available()); print('Device Count     :', torch.cuda.device_count()); print('Device Name      :', torch.cuda.get_device_name(0)); print('CUDA Version     :', torch.version.cuda); print('VRAM Total (GB)  :', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))"
  ```
  Ekspektasi: `True`, 1 device, `NVIDIA GeForce RTX 3070 Ti Laptop GPU`, `12.8`, `7.68`.

- [ ] **2.3 — Verifikasi dependensi inti**
  ```bash
  python -c "import numpy, pandas, scipy, sklearn, matplotlib, librosa, soundfile, tqdm; print('deps OK')"
  ```

- [ ] **2.4 — (Hanya jika 2.2 gagal) pasang ulang wheel CUDA**
  ```bash
  export UV_HTTP_TIMEOUT=300
  uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  uv pip install -r requirements.txt
  ```

- [ ] **2.5 — (Hanya jika venv rusak) buat ulang dari nol**
  ```bash
  rm -rf .venv
  uv venv .venv --python 3.11
  source .venv/bin/activate
  export UV_HTTP_TIMEOUT=300
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  uv pip install -r requirements.txt
  ```

**Definisi selesai Fase 2:** `torch.cuda.is_available() == True` dan semua dependensi ter-import tanpa error.

---

# 🔹 FASE 3 — Gate Import & Unit Test 8/8 (± 2 menit)

- [ ] **3.1 — Import paket `src` tanpa error**
  ```bash
  python -c "import src; print('src import OK')"
  ```

- [ ] **3.2 — Jalankan seluruh unit test**
  ```bash
  python tests/test_baseline.py 2>&1 | tee experiments/results/unit_test_deam.log
  ```
  Ekspektasi: `Ran 8 tests in ~1.3s` dan `OK`.

- [ ] **3.3 — Pastikan 8 pengujian yang dimaksud benar-benar lulus**
  | # | Test | Yang divalidasi |
  | :---: | :--- | :--- |
  | 01 | `test_01_deam_dataset_integrity` | 1.802 lagu, V/A ∈ [-1, 1] |
  | 02 | `test_02_preprocessing_pipeline` | 12 komponen PCA, skala [-1, 1] |
  | 03 | `test_03_kdtree_spatial_indexing` | >1.000 titik unik, 4 kuadran |
  | 04 | `test_04_stage1_waypoint_math` | Persamaan 5 Gaur |
  | 05 | `test_05_stage1_filtering_integrity` | anti-duplikasi & proteksi destinasi |
  | 06 | `test_06_stage2_distance_metrics` | Cosine/Euclidean/Manhattan |
  | 07 | `test_07_playlist_generation_constraints` | panjang N, unik, awal/akhir benar |
  | 08 | `test_08_metrics_bounds` | batas Smoothness & Evenness |

- [ ] **3.4 — Jangan lanjut jika ada test gagal.** Perbaiki dulu; jangan menimpa kegagalan dengan asumsi "nanti juga jalan".

**Definisi selesai Fase 3:** 8/8 test `OK`, log tersimpan.

---

# 🔹 FASE 4 — Verifikasi Layer Data & Normalisasi Emosi (± 2 menit)

- [ ] **4.1 — Muat anotasi statis DEAM dan cek jumlah + rentang**
  ```bash
  python - <<'PY'
  from src.deam_loader import load_or_generate_deam_dataset, get_default_deam_path
  print("DEAM path:", get_default_deam_path())
  df, cols = load_or_generate_deam_dataset()
  print("Jumlah lagu :", len(df))
  print("Jumlah fitur:", len(cols), cols)
  print("Rentang V   :", round(df.valence.min(), 4), "s/d", round(df.valence.max(), 4))
  print("Rentang A   :", round(df.arousal.min(), 4), "s/d", round(df.arousal.max(), 4))
  print("Contoh ID   :", df.track_id.head(3).tolist())
  PY
  ```
  Ekspektasi: 1802 lagu; V/A di dalam `[-1, 1]`; `track_id` berawalan `DEAM_`.
  **Jika `track_id` berawalan `TRK_` → path dataset salah, ulangi Fase 0.2.**

- [ ] **4.2 — Cek distribusi kuadran (bahan stratified split & pembahasan)**
  ```bash
  python - <<'PY'
  from src.deam_loader import load_or_generate_deam_dataset
  from src.split_utils import assign_emotion_quadrant
  df, _ = load_or_generate_deam_dataset()
  print(assign_emotion_quadrant(df).value_counts())
  print((assign_emotion_quadrant(df).value_counts(normalize=True) * 100).round(2))
  PY
  ```
  Ekspektasi mengikuti `split_summary.json`: Q3 Sad ±41%, Q1 Happy ±33%, Q4 Calm ±13%, Q2 Angry ±13%.

- [ ] **4.3 — (Opsional, jembatan ke model usulan) cek anotasi dinamis**
  ```bash
  python - <<'PY'
  from src.deam_loader import load_deam_dynamic_annotations
  v, a = load_deam_dynamic_annotations()
  print("Valence dynamic:", v.shape, "Arousal dynamic:", a.shape)
  print(v.columns.tolist()[:6])
  PY
  ```
  Baseline tidak memakai ini; hanya untuk memastikan dataset siap untuk penelitian lanjutan.

- [ ] **4.4 — Tentukan sumber fitur akustik yang dipakai (keputusan eksplisit)**
  - **Mode A (cepat, default sekarang):** fitur akustik sintetis deterministik (`random_state=42`) yang dikorelasikan dengan V/A. Cukup untuk menguji kebenaran algoritma, **tetapi bukan bukti akustik nyata**.
  - **Mode B (disarankan untuk Bab Hasil):** fitur akustik nyata dari 1.802 MP3 via Fase 5.
  Catat pilihan ini di log dan di naskah. Jangan mencampur hasil Mode A dan Mode B dalam satu tabel.

**Definisi selesai Fase 4:** 1802 lagu, ID `DEAM_*`, rentang [-1, 1], distribusi kuadran sesuai dokumen.

---

# 🔹 FASE 5 — (Opsional namun disarankan) Ekstraksi Fitur Akustik Nyata dari 1.802 MP3 (± 1–2 jam)

> **Peringatan penting:** `src/deam_extractor.py` menelan exception dan **diam-diam** menghasilkan fitur acak jika satu file gagal dibaca. Tidak ada tanda error di output. Karena itu validasi 5.2 wajib.
> **Ekstraksi tidak bisa resume.** Hasil baru ditulis di akhir. Jangan Ctrl+C di tengah jalan. Jika butuh aman, jalankan di `tmux`/`nohup`.

- [ ] **5.1 — Smoke test 5 file lebih dulu**
  ```bash
  mkdir -p /tmp/opencode
  python src/deam_extractor.py --limit 5 --output /tmp/opencode/deam_features_smoke.csv
  ```

- [ ] **5.2 — Validasi hasil smoke test (deteksi fallback acak)**
  ```bash
  python - <<'PY'
  import pandas as pd
  d = pd.read_csv('/tmp/opencode/deam_features_smoke.csv')
  print(d.shape)
  print(d[['mfcc_1','spectral_centroid','tempo_bpm','rms_energy']].describe().round(3))
  print("NaN:", int(d.isna().sum().sum()))
  print("song_id unik:", d.song_id.nunique())
  PY
  ```
  Cek kewajaran: `tempo_bpm` umumnya 60–180, `spectral_centroid` ratusan–ribuan Hz, `rms_energy` 0–1, tidak ada NaN. Jika nilai terlihat acak seragam tanpa pola, kemungkinan codec MP3 gagal (cek `ffmpeg`/`libsndfile`).

- [ ] **5.3 — Jalankan ekstraksi penuh di background dengan log**
  ```bash
  mkdir -p data/processed experiments/results
  nohup python src/deam_extractor.py \
    > experiments/results/extract_features_deam.log 2>&1 &
  echo $! > /tmp/opencode/extract_deam.pid
  ```
  Pantau: `tail -f experiments/results/extract_features_deam.log`
  Estimasi: 1–3 detik/lagu → **± 30–90 menit** untuk 1.802 lagu.

- [ ] **5.4 — Pastikan proses selesai normal**
  ```bash
  ps -p "$(cat /tmp/opencode/extract_deam.pid)" >/dev/null && echo "MASIH JALAN" || echo "SELESAI/BERHENTI"
  tail -5 experiments/results/extract_features_deam.log
  ```
  Ekspektasi log: `[+] Saved acoustic features for 1802 songs to: .../data/processed/deam_extracted_features.csv`

- [ ] **5.5 — Validasi CSV final (gerbang kualitas)**
  ```bash
  python - <<'PY'
  import pandas as pd
  p = 'data/processed/deam_extracted_features.csv'
  d = pd.read_csv(p)
  print("Shape:", d.shape)
  print("song_id unik:", d.song_id.nunique(), "| duplikat:", int(d.song_id.duplicated().sum()))
  print("NaN total:", int(d.isna().sum().sum()))
  assert len(d) == 1802, f"JUMLAH TIDAK 1802: {len(d)}"
  assert d.song_id.nunique() == 1802
  assert d.isna().sum().sum() == 0
  print("VALID")
  PY
  ```
  Jika jumlah < 1802, sebagian lagu gagal diekstrak → perbaiki dulu (jangan lanjut dengan data bolong).

- [ ] **5.6 — Konfirmasi auto-detect bekerja**
  ```bash
  python - <<'PY'
  from src.deam_loader import load_or_generate_deam_dataset
  df, cols = load_or_generate_deam_dataset()
  print("Fitur dimuat:", cols)
  PY
  ```
  Jika Fase 5 selesai, loader otomatis memakai CSV nyata (15 kolom fitur). Jika CSV dihapus, loader kembali ke fitur sintetis.

**Definisi selesai Fase 5:** `data/processed/deam_extracted_features.csv` berisi tepat 1.802 baris, 1.802 `song_id` unik, tanpa NaN; sumber fitur tercatat di log.

> Jika memilih **tidak** menjalankan Fase 5: lewati dan tulis di naskah bahwa fitur akustik yang dipakai adalah representasi sintetis deterministik. Jangan mengklaim hasil sebagai "fitur audio nyata".

---

# 🔹 FASE 6 — Verifikasi Pipeline Preprocessing (± 2 menit)

- [ ] **6.1 — Jalankan preprocessing dan catat explained variance**
  ```bash
  python - <<'PY'
  import numpy as np
  from src.deam_loader import load_or_generate_deam_dataset
  from src.preprocessing import GaurPreprocessor
  df, audio_cols = load_or_generate_deam_dataset()
  pre = GaurPreprocessor(n_pca_components=12, iqr_multiplier=5.5, n_bins=20)
  clean, pca_cols = pre.fit_transform(df, audio_feature_cols=audio_cols)
  print("Jumlah baris :", len(clean))
  print("Kolom PCA    :", pca_cols)
  print("Explained Var:", round(float(np.sum(pre.get_explained_variance_ratio())) * 100, 2), "%")
  print("Rentang PCA  :", round(clean[pca_cols].min().min(), 4), "s/d", round(clean[pca_cols].max().max(), 4))
  assert len(pca_cols) == 12
  assert clean[pca_cols].min().min() >= -1.0001
  assert clean[pca_cols].max().max() <= 1.0001
  print("PREPROCESSING VALID")
  PY
  ```
  Ekspektasi: 12 kolom, skala [-1, 1]. Explained variance tergantung Mode A/B; catat angkanya untuk naskah.

- [ ] **6.2 — Pastikan urutan transform benar (anti-leakage)**
  Baca `src/preprocessing.py`: `fit_transform` hanya boleh dipanggil pada Train (atau pada seluruh katalog untuk eksperimen retrieval baseline, dengan catatan metodologis). `transform` dipakai untuk Val/Test. Jangan memanggil `fit` pada Val/Test.

**Definisi selesai Fase 6:** 12 komponen PCA, skala [-1, 1], explained variance tercatat.

---

# 🔹 FASE 7 — Stratified Split & Verifikasi Anti-Data Leakage (± 5 menit)

- [ ] **7.1 — Cek manifest yang sudah ada**
  ```bash
  python - <<'PY'
  import pandas as pd, json
  tr = pd.read_csv('experiments/splits/deam_train_manifest.csv')
  va = pd.read_csv('experiments/splits/deam_val_manifest.csv')
  te = pd.read_csv('experiments/splits/deam_test_manifest.csv')
  print("Train/Val/Test:", len(tr), len(va), len(te), "| total:", len(tr)+len(va)+len(te))
  ids = [set(tr.song_id), set(va.song_id), set(te.song_id)]
  print("Irisan Train-Val:", len(ids[0] & ids[1]))
  print("Irisan Train-Test:", len(ids[0] & ids[2]))
  print("Irisan Val-Test:", len(ids[1] & ids[2]))
  assert len(ids[0] & ids[1]) == len(ids[0] & ids[2]) == len(ids[1] & ids[2]) == 0
  print("ZERO LEAKAGE OK")
  print(json.dumps(json.load(open('experiments/splits/split_summary.json')), indent=2)[:400])
  PY
  ```
  Ekspektasi: 1261 / 270 / 271, semua irisan 0.

- [ ] **7.2 — Regenerasi split (hanya jika Fase 5 mengubah dataset atau ingin split baru)**
  ```bash
  python -c "from src import load_or_generate_deam_dataset, stratified_split_deam, assign_emotion_quadrant, save_split_manifests; df, _ = load_or_generate_deam_dataset(); df['quadrant'] = assign_emotion_quadrant(df); tr, va, te = stratified_split_deam(df, 0.7, 0.15, 0.15); save_split_manifests(tr, va, te, 'experiments/splits'); print('Manifest split tersimpan:', len(tr), len(va), len(te))"
  ```
  Ulangi 7.1 setelah regenerasi. Catat perubahan angka bila ada.

- [ ] **7.3 — Pastikan split dipakai konsisten**
  Putuskan dan tulis di log: apakah eksperimen retrieval (single & batch) memakai seluruh 1.802 katalog atau hanya Train. Baseline Gaur melakukan retrieval pada katalog penuh; split 70/15/15 disiapkan untuk evaluasi model usulan. Jika memakai seluruh katalog, nyatakan eksplisit agar tidak dituduh leakage.

**Definisi selesai Fase 7:** 1261/270/271, irisan 0, keputusan penggunaan split tercatat.

---

# 🔹 FASE 8 — Single Run Experiment (Q3 Sad → Q1 Happy) di CUDA (± 1–3 menit)

- [ ] **8.1 — Jalankan eksperimen tunggal dengan log**
  ```bash
  python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7 \
    2>&1 | tee experiments/results/run_single_deam.log
  ```

- [ ] **8.2 — Verifikasi output konsol**
  Yang wajib muncul:
  - `[⚡ GPU ACTIVE] Using: NVIDIA GeForce RTX 3070 Ti Laptop GPU | CUDA: 12.8`
  - `Active Dataset: DEAM (1,802 Tracks) | Total Tracks: 1802`
  - `PCA 12 Components Fitted. Cumulative Explained Variance: ...%`
  - `KD-Tree indexed over ... unique coordinate points` (harus > 1.000)
  - Tabel 3 metrik (Cosine/Euclidean/Manhattan)
  - `[V] DEAM Baseline experiment execution completed successfully!`

- [ ] **8.3 — Verifikasi artefak dan asal dataset (gerbang anti-regresi)**
  ```bash
  ls -l experiments/results/metrics_comparison_deam.csv \
        experiments/results/baseline_cosine_deam_playlist.csv \
        experiments/results/baseline_deam_traversal_plot.png
  head -3 experiments/results/baseline_cosine_deam_playlist.csv
  ```
  **Wajib:** `track_id` berawalan `DEAM_`. Jika masih `TRK_*`, run memakai data sintetis → kembali ke Fase 0.2/4.1.

- [ ] **8.4 — Cek integritas playlist**
  ```bash
  python - <<'PY'
  import pandas as pd
  p = pd.read_csv('experiments/results/baseline_cosine_deam_playlist.csv')
  print("Panjang:", len(p), "| unik:", p.track_id.nunique())
  print("Step 0 :", p.iloc[0][['track_id','valence','arousal']].to_dict())
  print("Step N :", p.iloc[-1][['track_id','valence','arousal']].to_dict())
  assert len(p) == 12 and p.track_id.nunique() == 12
  assert p.iloc[0].valence < 0 and p.iloc[0].arousal < 0   # Q3
  assert p.iloc[-1].valence > 0 and p.iloc[-1].arousal > 0 # Q1
  print("PLAYLIST VALID")
  PY
  ```

- [ ] **8.5 — Catat metrik hasil ke log/naskah**
  Catat `Smoothness (PCC)`, `Evenness V-A Var`, `Evenness Audio Var`, `Mean V-A Step`, `Mean Audio Step` untuk ketiga metrik. Jangan menyalin angka dari dokumen lama — angka lama berasal dari run sintetis dan berbeda.

- [ ] **8.6 — Buka plot untuk inspeksi visual**
  `experiments/results/baseline_deam_traversal_plot.png` harus menampilkan lintasan diagonal dari kiri-bawah (Q3) ke kanan-atas (Q1) pada bidang Russell.

**Definisi selesai Fase 8:** log menunjukkan GPU aktif + dataset DEAM, 3 artefak baru, playlist 12 lagu unik `DEAM_*` dari Q3 ke Q1, metrik tercatat.

---

# 🔹 FASE 9 — Batch Quadrant Benchmark 540 Playlist (± 1–5 menit)

- [ ] **9.1 — Jalankan batch 6 rute**
  ```bash
  python experiments/run_quadrant_experiments.py \
    2>&1 | tee experiments/results/run_batch_deam.log
  ```
  Konfigurasi yang dijalankan skrip: 6 rute × K ∈ {5, 7, 15} × N ∈ {5, 12, 19} × 2 metrik × 5 trial = **540 evaluasi playlist**.

- [ ] **9.2 — Verifikasi jumlah baris dan kelengkapan rute**
  ```bash
  python - <<'PY'
  import pandas as pd
  d = pd.read_csv('experiments/results/batch_quadrant_experiments_deam.csv')
  print("Shape:", d.shape)
  print("Rute:", sorted(d.Route.unique()))
  print("K:", sorted(d.K_Neighbors.unique()), "| N:", sorted(d.Playlist_Length.unique()))
  print("Metrik:", sorted(d.Distance_Metric.unique()))
  print("Jumlah per rute:\n", d.Route.value_counts())
  print("NaN:\n", d.isna().sum()[d.isna().sum() > 0])
  assert len(d) == 540, f"HARUS 540 baris, dapat {len(d)}"
  print("BATCH VALID")
  PY
  ```
  **Catatan penting:** skrip menelan exception per trial (`except Exception: pass`). Jika baris < 540, ada trial gagal tanpa pesan. Penyebab umum: kuadran tujuan tidak punya cukup kandidat untuk N besar/K kecil. Jangan dianggap sukses sebelum 540.

- [ ] **9.3 — Cek metrik berada di rentang wajar**
  ```bash
  python - <<'PY'
  import pandas as pd
  d = pd.read_csv('experiments/results/batch_quadrant_experiments_deam.csv')
  print(d[['Smoothness_PCC','Evenness_VA_Var','Evenness_Audio_Var']].describe().round(6))
  assert d.Smoothness_PCC.between(-1.0001, 1.0001).all()
  assert (d.Evenness_VA_Var >= 0).all() and (d.Evenness_Audio_Var >= 0).all()
  print("METRIK VALID")
  PY
  ```

- [ ] **9.4 — Simpan tabel agregat untuk naskah**
  ```bash
  python - <<'PY'
  import pandas as pd
  d = pd.read_csv('experiments/results/batch_quadrant_experiments_deam.csv')
  agg = (d.groupby(['Route','Distance_Metric'])
           [['Smoothness_PCC','Evenness_VA_Var','Evenness_Audio_Var','Mean_VA_Step','Mean_Audio_Step']]
           .mean().round(6).reset_index())
  agg.to_csv('experiments/results/batch_quadrant_summary_deam.csv', index=False)
  print(agg.to_string(index=False))
  PY
  ```

**Definisi selesai Fase 9:** CSV batch tepat 540 baris, 6 rute, semua K/N/metrik lengkap, tanpa NaN, metrik valid, tabel agregat tersimpan.

---

# 🔹 FASE 10 — Validasi Hasil & Analisis untuk Naskah (± 30 menit)

- [ ] **10.1 — Bandingkan hasil dengan kriteria keberhasilan**
  | Metrik | Rentang | Arah optimal | Ekspektasi baseline DEAM |
  | :--- | :---: | :--- | :--- |
  | Smoothness (PCC) | [0, 1] | mendekati 1 | > 0.90 (dokumen lama 0.978–0.987, angka sintetis) |
  | Evenness V-A Var | ≥ 0 | mendekati 0 | kecil (orde 10⁻³–10⁻⁴) |
  | Evenness Audio Var | ≥ 0 | mendekati 0 | kecil |
  | Mean V-A Step | [0, √2] | tergantung N | ± 0.04–0.06 untuk N=12 |
  Jangan memaksa hasil baru sama dengan angka dokumen lama. Jika berbeda, jelaskan penyebabnya (fitur nyata vs sintetis, path dataset, seed).

- [ ] **10.2 — Bandingkan Cosine vs Euclidean vs Manhattan**
  Tentukan pemenang berdasarkan Smoothness dan Evenness, lalu tulis interpretasinya (bukan sekadar tabel).

- [ ] **10.3 — Analisis per rute kuadran**
  Identifikasi rute dengan Smoothness terendah / Evenness tertinggi. Biasanya rute vertikal/horizontal lebih sulit daripada diagonal. Kaitkan dengan kepadatan titik DEAM di tiap kuadran (Q3 41%, Q1 33%, Q2/Q4 ±13%).

- [ ] **10.4 — Cek efek K dan N**
  Buat tabel/pivot Smoothness vs K dan vs N. Jelaskan trade-off: K kecil → kandidat sedikit (bisa gagal), K besar → lebih beragam tapi kurang halus.

- [ ] **10.5 — Simpan bukti eksekusi**
  ```bash
  mkdir -p experiments/results/archive_$(date +%Y%m%d_%H%M)
  cp experiments/results/*.csv experiments/results/*.png experiments/results/*.log \
     experiments/results/archive_$(date +%Y%m%d_%H%M)/ 2>/dev/null || true
  ls -lh experiments/results/archive_*
  ```

- [ ] **10.6 — Catat provenance run (wajib untuk reproduksibilitas)**
  Tulis di log/naskah: commit hash, mode fitur (nyata/sintetis), jumlah lagu, K, N, metrik, seed, versi torch/CUDA, tanggal.

**Definisi selesai Fase 10:** tabel perbandingan, analisis per rute, analisis K/N, arsip hasil, dan provenance tercatat.

---

# 🔹 FASE 11 — Reproducibility, Git, dan Penutupan (± 15 menit)

- [ ] **11.1 — Pastikan tidak ada konflik & file besar tidak ikut ter-stage**
  ```bash
  grep -rn '^<<<<<<< \|^=======$\|^>>>>>>> ' --exclude-dir=.git --exclude-dir=.venv . || echo "BERSIH"
  git status --short
  git check-ignore -v .venv Dataset data/processed experiments/results/batch_quadrant_experiments_deam.csv
  ```
  Semua path besar harus ter-ignore.

- [ ] **11.2 — Perbarui dokumentasi agar sesuai kondisi laptop**
  - `TUTORIAL_RUN_CUDA.md`: path `Dataset/DEAM` (bukan `EXPERIMENT/Dataset`), angka hasil aktual, mode fitur.
  - `PANDUAN_VALIDASI_DAN_TESTING.md`: ganti klaim hasil lama bila sudah diregenerasi.
  - `README.md`: bagian setup Linux + split.
  - `experiments/configs/params_deam.json`: selaraskan dengan batch yang benar-benar dijalankan (batch skrip memakai 5 trial, K {5,7,15}, N {5,12,19}, 2 metrik — bukan isi JSON saat ini).

- [ ] **11.3 — Commit kode + dokumen + log teks (bukan artefak besar)**
  ```bash
  git add src experiments tests README.md PANDUAN_VALIDASI_DAN_TESTING.md \
          TUTORIAL_RUN_CUDA.md Experiment-Baseline-KNN-Gaur-Todolist.md todolist.md \
          requirements.txt pyproject.toml .gitignore
  git commit -m "feat: verified DEAM baseline (single + 540 batch) on local CUDA RTX 3070 Ti"
  git log --oneline -5
  ```

- [ ] **11.4 — Push ke GitHub**
  ```bash
  git push origin main
  ```
  Jika artefak CSV/PNG ingin ikut dipublikasikan (Open Data), paksa tambah secara eksplisit setelah mengecilkan ukuran:
  ```bash
  git add -f experiments/results/metrics_comparison_deam.csv \
               experiments/results/baseline_cosine_deam_playlist.csv \
               experiments/results/batch_quadrant_experiments_deam.csv \
               experiments/results/batch_quadrant_summary_deam.csv \
               experiments/results/baseline_deam_traversal_plot.png
  ```
  Jangan pernah `git add -f` file MP3, `.venv`, atau `data/processed/*.csv` berukuran besar tanpa pertimbangan.

- [ ] **11.5 — Bersihkan stash lama setelah yakin**
  ```bash
  git stash list
  # baru jalankan setelah commit & push sukses dan hasil tervalidasi:
  git stash drop stash@{0}
  ```

**Definisi selesai Fase 11:** repo bersih, commit terdorong, dokumentasi sesuai kondisi nyata, tidak ada file besar ter-commit.

---

# ✅ FASE 12 — Checklist Penerimaan Akhir (Definition of Done)

Centang semuanya sebelum menyatakan eksperimen selesai:

- [ ] Tidak ada merge conflict marker di seluruh repo.
- [ ] `torch.cuda.is_available() == True` dan GPU name benar.
- [ ] Dataset terdeteksi 1.802 lagu dengan `track_id` `DEAM_*`.
- [ ] Unit test 8/8 `OK`, log tersimpan.
- [ ] Sumber fitur akustik dinyatakan eksplisit (nyata / sintetis) dan konsisten di semua hasil.
- [ ] Preprocessing menghasilkan 12 komponen PCA dalam skala [-1, 1]; explained variance tercatat.
- [ ] Split 1261/270/271 tanpa irisan (jika dipakai).
- [ ] `run_experiment.py` sukses, playlist 12 lagu unik Q3 → Q1, plot valid.
- [ ] `run_quadrant_experiments.py` menghasilkan **tepat 540 baris**, 6 rute lengkap, tanpa NaN.
- [ ] Tabel agregat per rute/metrik tersimpan.
- [ ] Provenance run (commit, seed, versi, mode fitur, tanggal) tercatat.
- [ ] Commit + push selesai; file besar tetap ter-ignore.
- [ ] Angka di dokumen panduan sudah disinkronkan dengan hasil run terbaru.

---

# 🧰 Lampiran A — Cheat Sheet Perintah

```bash
# Setup
cd ~/TA-123140205/Code/Baseline-KNN-Gaur
source .venv/bin/activate

# Cek GPU
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Gate kualitas
python -m py_compile src/*.py experiments/*.py tests/test_baseline.py
python tests/test_baseline.py

# Data
python -c "from src.deam_loader import load_or_generate_deam_dataset as f; d,c=f(); print(len(d), len(c), d.track_id.head(3).tolist())"

# Ekstraksi fitur nyata (opsional)
python src/deam_extractor.py --limit 5 --output /tmp/opencode/deam_features_smoke.csv
nohup python src/deam_extractor.py > experiments/results/extract_features_deam.log 2>&1 &

# Split
python -c "from src import load_or_generate_deam_dataset, stratified_split_deam, assign_emotion_quadrant, save_split_manifests; df,_=load_or_generate_deam_dataset(); df['quadrant']=assign_emotion_quadrant(df); tr,va,te=stratified_split_deam(df,.7,.15,.15); save_split_manifests(tr,va,te,'experiments/splits'); print(len(tr),len(va),len(te))"

# Eksperimen
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7
python experiments/run_quadrant_experiments.py
```

---

# 🧯 Lampiran B — Troubleshooting Cepat

| Gejala | Penyebab paling mungkin | Tindakan |
| :--- | :--- | :--- |
| `IndentationError` / `SyntaxError` di `src/deam_loader.py` | Merge conflict belum diselesaikan | Fase 0.2 |
| `track_id` hasil `TRK_*`, bukan `DEAM_*` | Path DEAM tidak ketemu → fallback sintetis | Fase 0.2, lalu Fase 4.1 |
| `CUDA Available: False` | Torch dari PyPI CPU, atau driver tidak ter-load | Fase 2.4, cek `lsmod \| grep nvidia` |
| `ModuleNotFoundError: src` | Dijalankan dari folder salah / venv tidak aktif | `cd` root proyek, `source .venv/bin/activate` |
| Batch CSV < 540 baris | Exception per trial ditelan `except: pass` (kuadran jarang, N besar, K kecil) | Kurangi N / naikkan K, atau instrumentasi ulang; jangan klaim sukses |
| Ekstraksi audio selesai < 1802 baris | Sebagian MP3 gagal dibaca / codec | Cek log, pastikan `ffmpeg`/`libsndfile` ada, ulangi |
| Fitur audio terlihat acak | Fallback senyap di `deam_extractor.py` | Fase 5.2; pertimbangkan patch agar `raise` saat gagal |
| `nvidia-smi` error | Driver belum terpasang/ter-load | `sudo pacman -S --needed nvidia-utils nvidia-settings`, reboot |
| Instalasi torch timeout | Wheel CUDA ±5 GB | `export UV_HTTP_TIMEOUT=300`, ulangi (cache melanjutkan) |
| OOM VRAM | Proses lain memakai GPU | `nvidia-smi`; baseline hanya butuh <100 MB |

---

# ⏱️ Lampiran C — Estimasi Waktu

| Fase | Estimasi | Catatan |
| :--- | :---: | :--- |
| 0 — Resolve conflict | 10 menit | Blocker, kerjakan pertama |
| 1 — Preflight | 2 menit | — |
| 2 — Environment | 5 menit | Sudah terpasang; hanya verifikasi |
| 3 — Unit test | 2 menit | 8/8 |
| 4 — Verifikasi data | 2 menit | — |
| 5 — Ekstraksi audio nyata | 30–90 menit | Opsional; tanpa resume |
| 6 — Preprocessing | 2 menit | — |
| 7 — Split | 5 menit | — |
| 8 — Single run | 1–3 menit | Dominan waktu load/import |
| 9 — Batch 540 | 1–5 menit | Komputasi kecil, GPU tidak dominan |
| 10 — Analisis | 30 menit | Paling banyak berpikir, bukan komputasi |
| 11 — Git & dokumentasi | 15 menit | — |
| **Total tanpa ekstraksi** | **± 1,5 jam** | Termasuk analisis |
| **Total dengan ekstraksi** | **± 2,5–3,5 jam** | Tergantung durasi dekode MP3 |

---

*Dokumen ini dibuat untuk kondisi nyata laptop ini per 19 November 2026 dan sengaja menandai blocker, artefak stale, serta jebakan fallback senyap agar tidak ada langkah yang terlewat.*
