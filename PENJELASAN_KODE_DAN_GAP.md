# Penjelasan Menyeluruh Kode & Gap Penelitian

> **Objek:** `~/TA-123140205/Code/Baseline-KNN-Gaur/`
> **Referensi ilmiah:** Shaurya Gaur, *"Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"* (Master's Thesis, University of Victoria, 2023 / Springer EvoMUSART 2024)
> **Dataset:** DEAM — 1.802 klip audio 45 detik, anotasi emosi statis + dinamis (2 Hz)
> **Status kode:** replikasi baseline sudah berjalan end-to-end, **tanpa training model**
> **Dokumen ini:** menjelaskan apa yang *sebenarnya* terjadi saat kode dijalankan, kelemahan implementasinya, angka hasilnya, dan di mana celah penelitian (research gap) yang bisa diisi TA.

---

## 1. Ringkasan Eksekutif

Repo ini adalah **replikasi baseline**, bukan model AI yang dilatih. Isinya: satu pipeline yang (1) memuat anotasi emosi DEAM, (2) mereduksi 15 fitur akustik menjadi 12 dimensi PCA, (3) membangun indeks KD-Tree di ruang emosi 2D, lalu (4) menyusun playlist dengan **interpolasi garis lurus + pencarian tetangga terdekat + pemeringkatan jarak akustik**.

Semua keputusan terjadi **saat playlist diminta (query-time)**. Tidak ada satu pun bobot model yang diperbarui, tidak ada loss, tidak ada epoch, tidak ada checkpoint.

| Pertanyaan | Jawaban |
| :--- | :--- |
| Ada training? | **Tidak.** Nol `optimizer`, nol `backward()`, nol loop epoch |
| Pakai ML? | KNN (lazy learner) + PCA. Tidak ada parameter yang dioptimasi dari data |
| GPU dipakai? | Hanya untuk menghitung jarak 7 vektor 12D per langkah. VRAM tercatat **0.00 MB** |
| Hasilnya apa? | Playlist 12 lagu Q3→Q1 + 540 evaluasi batch di 6 rute kuadran |
| Sebagai apa di TA? | **Pembanding (baseline)** untuk model usulan yang belum ada |

Kalimat yang aman untuk naskah:

> *"Baseline Gaur (2023) direplikasi sebagai metode retrieval dua tahap berbasis KNN dan KD-Tree pada dataset DEAM, tanpa proses pelatihan parameter, dan dievaluasi dengan metrik Smoothness serta Evenness."*

Kalimat yang **salah** dan berisiko di sidang:

> ~~"Kami melatih model KNN..."~~ — tidak ada yang dilatih.
> ~~"Model AI baseline mencapai akurasi 99%..."~~ — tidak ada akurasi, dan 99% itu explained variance PCA, bukan performa.

---

## 2. Peta File: Apa Fungsi Sebenarnya

| File | Baris | Fungsi nyata | Peran di pipeline |
| :--- | ---: | :--- | :--- |
| `src/deam_loader.py` | 151 | Baca 2 CSV anotasi statis, normalisasi V/A dari [1,9] ke [-1,1], tempel fitur akustik | Input data |
| `src/deam_extractor.py` | 139 | Ekstraksi 15 fitur librosa dari MP3, fallback acak kalau gagal | Pembuat fitur |
| `src/preprocessing.py` | 131 | Clip outlier IQR → Yeo-Johnson → PCA 12D → MinMax [-1,1] | Transformasi |
| `src/dataset.py` | 142 | Lookup track, pemetaan koordinat unik → daftar lagu, bangun KD-Tree; juga generator dataset sintetis | Indeks |
| `src/stage1_knn.py` | 97 | Hitung waypoint Persamaan 5; query KD-Tree untuk K kandidat; buang lagu terpakai & lagu tujuan | Tahap 1 |
| `src/stage2_selector.py` | 127 | Hitung jarak akustik kandidat ke lagu terakhir; pilih minimum (PyTorch CUDA / SciPy fallback) | Tahap 2 |
| `src/playlist_generator.py` | 120 | Loop langkah 1..N-2, panggil Tahap 1 + Tahap 2, tempel lagu tujuan di akhir | Orkestrator |
| `src/metrics.py` | 110 | Smoothness = \|PCC(V,A)\|; Evenness = Var(RMSE langkah) | Evaluasi |
| `src/split_utils.py` | 197 | Stratified split 70/15/15 per kuadran + split katalog/query + 5-fold | Artefak (belum dipakai) |
| `src/utils_cuda.py` | 63 | Pilih device, cetak VRAM, konversi tensor | Utilitas |
| `experiments/run_experiment.py` | 178 | Satu playlist Q3→Q1 untuk 3 metrik + plot | Eksperimen A |
| `experiments/run_quadrant_experiments.py` | 136 | 540 playlist (6 rute × K × N × 2 metrik × 5 trial) | Eksperimen B |
| `tests/test_baseline.py` | 162 | 8 unit test integritas data, matematika, dan constraint | Gate kualitas |
| `wandb-set-up.py` | 30 | **Template contoh wandb** (CIFAR-100, CNN, acc/loss palsu) | ⚠️ Lihat §6.12 |

---

## 3. Apa yang Sebenarnya Terjadi Saat Dijalankan

### 3.1 Alur data

```
Dataset/DEAM/DEAM_audio/MEMD_audio/*.mp3        (1.802 file, 45 detik)
        │  deam_extractor.py (librosa)
        ▼
data/processed/deam_extracted_features.csv      (1.802 × 16: 15 fitur + song_id)
        │  deam_loader.py + 2 CSV anotasi statis
        ▼
DataFrame 1.802 baris: track_id, valence, arousal, 15 fitur akustik
        │  GaurPreprocessor.fit_transform()
        ▼
DataFrame 1.802 baris: track_id, V, A, pca_0 .. pca_11   (semua di [-1, 1])
        │  SongDatasetClass
        ▼
KD-Tree atas 1.113 koordinat (V,A) unik + lookup lagu per koordinat
        │  TwoStagePlaylistGenerator
        ▼
Playlist N lagu  →  metrics.py  →  CSV + PNG
```

### 3.2 Langkah 1 — Data emosi

- Anotasi statis DEAM berisi rata-rata penilaian sukarelawan per lagu, skala Likert 1–9.
- Normalisasi: `V = (valence_mean − 5) / 4`, `A = (arousal_mean − 5) / 4` → rentang [-1, 1].
- 1.802 lagu. Distribusi kuadran Russell: **Q3 Sad 742 (41.18%)**, **Q1 Happy 594 (32.96%)**, **Q4 Calm 237 (13.15%)**, **Q2 Angry 229 (12.71%)**.
- Ketimpangan ini penting: kuadran Q2/Q4 hanya ~13% katalog, sehingga rute yang melewatinya lebih sulit.

### 3.3 Langkah 2 — Fitur akustik

`deam_extractor.py` mengambil **45 detik pertama** tiap MP3 (22050 Hz) lalu menghitung:

| Kelompok | Fitur | Jumlah |
| :--- | :--- | ---: |
| MFCC | `mfcc_1` … `mfcc_6` (dari 8 yang dihitung, 2 dibuang) | 6 |
| Spektral | `spectral_centroid`, `spectral_rolloff`, `spectral_flatness`, `zero_crossing_rate`, `rms_energy` | 5 |
| Ritmis | `tempo_bpm` | 1 |
| Kromatik | `chroma_1` … `chroma_3` (dari 12, 9 dibuang) | 3 |
| **Total** | | **15** |

Semuanya **nilai rata-rata global** — dinamika temporal dalam 45 detik hilang. Ini bukan pilihan ilmiah, ini simplifikasi implementasi (lihat §6.5).

### 3.4 Langkah 3 — Preprocessing

Empat tahap, di-`fit` pada seluruh 1.802 lagu:

1. **Clip outlier IQR 5.5** — nilai di luar `Q1 − 5.5·IQR` / `Q3 + 5.5·IQR` dipotong ke batas.
2. **Yeo-Johnson** — power transform agar distribusi fitur lebih normal (menstabilkan variance).
3. **PCA 12 komponen** — 15 dimensi → 12 dimensi. Explained variance terukur: **99.10%** (Mode B, fitur nyata).
4. **MinMax [-1, 1]** — menyamakan skala dengan ruang emosi.

Yang perlu dicatat: **`fit_transform` dipanggil pada seluruh katalog**, bukan hanya train. Untuk retrieval ini wajar (katalog = database), tetapi tidak boleh diklaim sebagai eksperimen supervised tanpa leakage. `transform()` sudah tersedia dan benar (memakai parameter train), hanya belum dipakai di skrip eksperimen.

### 3.5 Langkah 4 — Indeks KD-Tree

- Koordinat (V,A) dibulatkan 4 desimal → **1.113 titik unik** dari 1.802 lagu (banyak lagu berbagi koordinat karena anotasi dibulatkan ke 2 desimal).
- KD-Tree di-fit pada 1.113 titik itu, bukan 1.802 lagu.
- `coord_to_track_ids` memetakan satu titik → daftar lagu.

Konsekuensi: query mengembalikan **titik**, lalu titik diterjemahkan menjadi lagu **sesuai urutan penyimpanan**. Urutan kandidat untuk lagu yang berbagi koordinat bersifat arbitrer (urutan dict), bukan berdasarkan preferensi apa pun.

### 3.6 Langkah 5 — Pemilihan lagu awal & tujuan

`get_song_in_quadrant(q, seed)` memilih acak satu lagu dari kuadran, dengan filter `±0.1` (bukan `0`). Fungsi ini memanggil `np.random.seed(seed)` — **mengubah RNG global**, bukan `RandomState` lokal. Deterministik, tapi efek sampingnya menjalar ke seluruh proses.

Di batch, `seed = 1000 + 17·trial` untuk asal dan `seed + 999` untuk tujuan. Efeknya bagus: **pasangan asal-tujuan sama untuk semua K/N/metrik pada trial yang sama**, sehingga perbandingan bersifat *paired* — meskipun belum dianalisis dengan uji statistik berpasangan.

### 3.7 Langkah 6 — Loop Tahap 1: interpolasi waypoint (Persamaan 5)

Untuk setiap langkah `L = 1 … N−2`:

```
S_T = S_L + (S_N − S_L) / (N − L)
```

- `S_L` = koordinat lagu terakhir yang sudah masuk playlist
- `S_N` = koordinat lagu tujuan (tetap)
- `N` = panjang playlist

Sifatnya: **garis lurus** dari posisi sekarang ke tujuan, dibagi sisa langkah. Jika lagu bergerak persis ke waypoint, lintasan akan menjadi garis lurus sempurna. Inilah sebabnya angka Smoothness tinggi — target desainnya memang garis lurus.

Lalu KD-Tree mencari titik terdekat, mengambil lagu-lagunya, dan menyaring:
- lagu yang sudah masuk playlist (`visited_track_ids`),
- lagu tujuan (tidak boleh muncul sebelum langkah N).

Jika kandidat kurang dari K, query diperluas ke seluruh 1.113 titik.

### 3.8 Langkah 7 — Tahap 2: re-ranking akustik

Dari K kandidat emosi, pilih **satu** yang jarak fitur akustiknya paling dekat dengan **lagu terakhir** (bukan dengan tujuan, bukan dengan seluruh lintasan):

```
pilih argmin_k  d(audio_vector_last, audio_vector_kandidat)
d ∈ { cosine, euclidean, manhattan }
```

Ini **greedy dan myopic**: hanya melihat satu langkah ke belakang. Tidak ada pertimbangan apakah pilihan ini membuat sisa lintasan lebih sulit, dan tidak ada model yang belajar dari pengalaman.

Perhitungan jarak dilakukan di GPU (7 vektor 12D per langkah) — secara komputasi tidak signifikan.

### 3.9 Langkah 8 — Penutup playlist

Lagu tujuan **selalu ditempel paksa** di indeks terakhir. Artinya:

- kondisi awal & akhir playlist dijamin benar **oleh konstruksi**, bukan oleh kemampuan algoritma,
- Smoothness/Evenness sebagian merupakan konsekuensi endpoint yang sudah dipatok.

### 3.10 Langkah 9 — Evaluasi

Dua metrik, keduanya **intrinsik** (tidak butuh ground truth):

**Smoothness (PCC):**
```
ρ = |corr(V_playlist, A_playlist)|
```
Mengukur seberapa lurus lintasan di bidang V-A. Jika `std(V) < 1e-6` atau `std(A) < 1e-6` → **dikembalikan 1.0** (kasus lintasan vertikal/horizontal sempurna).

**Evenness:**
```
step_k  = ||S_{k+1} − S_k||₂ / √D
Evenness = Var(step_k)
```
Semakin kecil → jarak antar lagu semakin seragam. Dihitung dua kali: di ruang emosi 2D dan ruang audio 12D.

Catatan penting: keduanya **tidak mengukur relevansi musikal atau kepuasan pengguna**. Playlist yang "halus secara matematis" belum tentu enak didengar.

---

## 4. Apa yang Bukan Training (dan Buktinya)

| Bukti | Hasil |
| :--- | :--- |
| `grep -r "optimizer\|backward()\|loss\|epoch\|DataLoader"` di `src/` | **0 hasil** |
| VRAM saat menjalankan single run & batch | `Allocated: 0.00 MB \| Reserved: 0.00 MB` |
| Checkpoint / bobot tersimpan | Tidak ada (`*.pt`, `*.pth` di-ignore dan tidak pernah dibuat) |
| Waktu eksekusi | Single run ~1 detik, batch 540 ~10 detik (bukan jam) |
| Kurva loss | Tidak ada |

**Soal "KNN itu ML"**: secara taksonomi KNN termasuk *supervised, instance-based, lazy learning*. "Lazy" berarti fase training-nya kosong — "belajar" = menyimpan data + membangun indeks. Tidak ada parameter yang dioptimasi. Jadi menyebut *"metode KNN"* benar; menyebut *"melatih model KNN"* tidak.

**Soal `fit` di sklearn**: `PowerTransformer.fit`, `PCA.fit`, `MinMaxScaler.fit` hanya menghitung statistik deskriptif (lambda, eigenvector, min/max). Itu **estimasi parameter transformasi**, bukan pembelajaran dari error. Tidak ada gradient.

---

## 5. Hasil yang Sudah Terverifikasi (Mode B — Fitur Nyata)

### 5.1 Single run (N=12, K=7, Q3→Q1)

Playlist: `DEAM_2022` (V=-0.300, A=-0.150) → `DEAM_1879` (V=0.100, A=0.350), 12 lagu unik.

| Metrik | Smoothness (PCC) | Evenness V-A Var | Evenness Audio Var | Mean V-A Step | Mean Audio Step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Cosine | **0.995043** | 0.000245 | 0.004586 | 0.042695 | 0.206768 |
| Euclidean | **0.992504** | 0.000288 | 0.002053 | 0.042230 | 0.191666 |
| Manhattan | **0.983973** | 0.000312 | 0.004336 | 0.043303 | 0.201044 |

### 5.2 Batch 540 (6 rute × K{5,7,15} × N{5,12,19} × {cosine,euclidean} × 5 trial)

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

Kesimpulan yang bisa diklaim:
1. Rute diagonal paling halus; rute yang melewati kuadran minoritas (Q2/Q4) paling kasar.
2. Semakin panjang playlist dan semakin besar K, semakin tidak halus.
3. Cosine vs Euclidean tidak ada pemenang mutlak (3 rute vs 3 rute; rata-rata 0.8516 vs 0.8432) — ini lebih jujur daripada mengklaim "Cosine selalu menang".

---

## 6. Kelemahan & Jebakan Implementasi yang Wajib Kamu Tahu

Bagian ini bukan untuk menjatuhkan kode, tapi supaya kamu tidak terkejut saat penguji membacanya.

### 6.1 "KBinsDiscretizer 20 bins" tidak benar-benar terjadi

`discretize_extreme_outliers()` membuat `KBinsDiscretizer`, memanggil `fit_transform`, menyimpan hasilnya ke variabel `discretized` — **lalu tidak memakainya**. Yang benar-benar diterapkan hanya `np.clip` ke batas IQR.

- README dan PANDUAN menyebut "IQR Discretizer → 20 bins".
- Kenyataan: hanya clipping.
- **Risiko:** penguji membaca kode, menemukan `discretized` tak terpakai, lalu mempertanyakan klaim metodologi.
- **Opsi:** (a) hapus klaim "discretizer" dan tulis "outlier clipping IQR 5.5", atau (b) benar-benar terapkan binning. (b) mengubah angka, (a) lebih aman.

### 6.2 Smoothness = |PCC| bukan metrik "kehalusan" yang murni

- Lintasan yang lurus diagonal **otomatis** PCC ≈ 1 — dan justru itu yang diperintahkan Persamaan 5. Jadi metrik ini sebagian mengukur "apakah algoritma mengikuti rumusnya sendiri", bukan kualitas musik.
- Jika V atau A hampir konstan, fungsi mengembalikan **1.0** → rute vertikal/horizontal bisa terlihat "sempurna" karena alasan teknis.
- PCC tidak peduli urutan lagu mana yang dipilih, hanya posisi koordinatnya.
- **Implikasi:** angka 0.99 bukan bukti playlist bagus. Harus disampaikan sebagai *adherence to the intended trajectory*, bukan kualitas persepsi.

### 6.3 Endpoint dipaksa, jadi sebagian hasil "gratis"

Lagu tujuan ditempel di akhir tanpa proses seleksi. Smoothness tinggi bisa terjadi walau lagu-lagu tengahnya dipilih buruk. Tidak ada evaluasi terpisah: "apakah lagu ke-N memang layak menjadi tujuan?"

### 6.4 Tahap 2 greedy satu langkah

Pemilihan hanya membandingkan dengan **lagu terakhir**. Tidak ada look-ahead, tidak ada optimasi global, tidak ada penalti jika sisa lintasan menjadi mustahil. Ini sumber utama penurunan performa di N besar (tabel §5.2).

### 6.5 Fitur akustik membuang dinamika temporal

- Hanya 45 detik pertama.
- Hanya nilai rata-rata (mean) → tidak ada variance, tidak ada perubahan antar bagian lagu.
- 6 dari 8 MFCC dipakai, 3 dari 12 chroma dipakai — pembatasan arbitrer.
- Padahal DEAM menyediakan anotasi **2 Hz** (500 ms) sepanjang lagu: `valence.csv` (1802 × 1224) dan `arousal.csv` (1802 × 1225). Informasi ini ada, tapi **tidak dipakai sama sekali** oleh baseline.

### 6.6 Fitur sintetis bisa menyamar sebagai nyata (dua tempat)

1. `deam_extractor.py` menelan exception lalu mengembalikan **angka acak** tanpa peringatan. Satu file MP3 rusak = satu baris "fitur" palsu yang tak bisa dibedakan dari yang asli.
2. `deam_loader.py` membuat 15 fitur sintetis deterministik bila CSV tidak ada.

Sudah dimitigasi (CSV nyata ada, 0 NaN, distribusi diperiksa), tetapi mekanismenya masih ada di kode. **Saran:** tambahkan `raise` atau kolom penanda `source` agar tidak bisa tertukar lagi.

### 6.7 `except Exception: pass` di batch

`run_quadrant_experiments.py` menelan semua error per trial. Jika ada yang gagal, kamu hanya melihat jumlah baris < 540 **tanpa pesan apa pun**. Saat ini 540/540 aman, tapi ke depan bisa menipu.

### 6.8 Split 70/15/15 adalah artefak mati

`split_utils.py` menghasilkan manifest train/val/test (1.261 / 270 / 271) dan anti-leakage-nya benar. Tapi **tidak ada skrip eksperimen yang memakainya**. Retrieval memakai seluruh 1.802 lagu.

Ini bukan kesalahan (memang perilaku Gaur), tapi harus dinyatakan eksplisit di Bab Metodologi — kalau tidak, akan terlihat seperti split yang dibuat lalu dilupakan.

### 6.9 Definisi kuadran tidak konsisten

- `split_utils.assign_emotion_quadrant`: batas **0** (`v > 0`, `a > 0`, dst.)
- `dataset.get_song_in_quadrant`: batas **±0.1** (ada zona netral)
- Distribusi kuadran di dokumen (742/594/237/229) dihitung dengan batas 0, sedangkan pemilihan lagu eksperimen memakai ±0.1.

Selisihnya kecil, tapi dua definisi untuk hal yang sama adalah celah pertanyaan.

### 6.10 Batch hanya Cosine & Euclidean

Manhattan hanya diuji di single run. `params_deam.json` (lama) mencantumkan 3 metrik, sehingga dokumen dan kode sempat tidak sinkron. Sudah disinkronkan, tapi klaim "ketiga metrik diuji pada 540 playlist" tetap salah — hanya 2.

### 6.11 Mean V-A Step tidak sebanding lintas N

Rata-rata langkah bergantung pada panjang lintasan: N=5 punya langkah besar, N=19 kecil. Membandingkan "Mean Step" antar-N tanpa normalisasi adalah kesalahan metodologis. Gunakan Smoothness/Evenness (yang sudah dinormalisasi) untuk perbandingan lintas N.

### 6.12 ⚠️ `wandb-set-up.py` adalah template kosong dengan metrik palsu

Isinya template resmi wandb:

```python
config={"learning_rate": 0.02, "architecture": "CNN",
        "dataset": "CIFAR-100", "epochs": 10}
for epoch in range(2, epochs):
    acc  = 1 - 2**-epoch - random.random()/epoch - offset
    loss = 2**-epoch + random.random()/epoch + offset
    run.log({"acc": acc, "loss": loss})
```

Dan folder `wandb/run-20261119_172848-lfbhhn5f/` **sudah berisi run tersimpan** dengan `acc: 0.796`, `loss: 0.151`, dataset tertulis `CIFAR-100`, arsitektur `CNN`.

Bahayanya:
- Dashboard wandb terlihat seperti "training CNN di CIFAR-100" — **tidak ada hubungannya dengan TA kamu**.
- Jika ikut ditunjukkan atau ikut ter-commit, ini bisa dianggap klaim palsu.

**Tindakan yang disarankan:** hapus `wandb-set-up.py` dan folder `wandb/` dari repo, atau ganti dengan logging yang benar-benar mencatat parameter eksperimen retrieval (K, N, metrik, metrik evaluasi) — bukan `acc`/`loss` fiktif.

---

## 7. Gap Penelitian

Bagian ini yang paling penting untuk TA. Setiap gap ditulis sebagai: *kondisi di baseline* → *mengapa itu celah* → *apa yang bisa kamu lakukan*.

### Gap 1 — Emosi statis vs emosi dinamis (gap terbesar)

**Kondisi:** Baseline memakai satu nilai V/A rata-rata per lagu. Lagu 45 detik dianggap satu titik emosi.
**Celah:** Musik berubah emosi sepanjang lagu. Gaur memakai rata-rata karena dataset Deezer-nya memang hanya punya rata-rata (disebut sendiri di `PANDUAN_VALIDASI_DAN_TESTING.md` §Q1). **DEAM kamu punya anotasi 2 Hz (500 ms) sepanjang lagu** — 1224–1225 kolom waktu per lagu.
**Peluang:** Model yang memprediksi atau memanfaatkan **kurva V/A dinamis**, bukan titik. Contoh rumusan: *"Bagaimana memanfaatkan anotasi emosi dinamis DEAM untuk menghasilkan playlist yang halus secara temporal, bukan hanya secara rata-rata?"*
**Dampak:** Ini kontribusi paling jelas dan paling mudah dipertahankan, karena datanya sudah ada dan baseline secara eksplisit tidak memakainya.

### Gap 2 — Pemilihan lagu tidak "belajar", hanya greedy

**Kondisi:** Tahap 2 memilih kandidat dengan jarak akustik minimum ke lagu sebelumnya (1 langkah ke belakang), tanpa look-ahead.
**Celah:** Keputusan lokal tidak menjamin kualitas lintasan global; inilah sebabnya N=19 & K=15 turun ke 0.76.
**Peluang:** Formulasikan sebagai **sequential decision / planning**:
- beam search / dynamic programming atas kandidat (tanpa training, tapi bisa mengalahkan greedy),
- model seq2seq (Transformer/LSTM) yang menghasilkan urutan track ID atau waypoint adaptif,
- RL dengan reward = Smoothness + Evenness + penalti pelanggaran endpoint.
**Dampak:** Perbandingan langsung "greedy vs learned planner" sangat kuat untuk Bab Hasil.

### Gap 3 — Metrik tidak punya ground truth

**Kondisi:** Smoothness/Evenness hanya mengukur sifat geometris lintasan. Tidak ada acuan "playlist yang benar".
**Celah:** Tidak ada jawaban untuk pertanyaan penguji: *"0.995 itu bagus menurut siapa?"* Tidak ada baseline pembanding (random walk, urutan berdasarkan popularity, urutan acak dalam kuadran).
**Peluang:**
- tambahkan baseline trivial: random, sorted-by-valence, sorted-by-arousal, nearest-neighbor tanpa waypoint;
- uji signifikansi (Wilcoxon signed-rank) antara metode — data batch sudah *paired* per trial, tinggal dipakai;
- (opsional, paling kuat) studi persepsi kecil: 10–15 responden menilai 20 playlist.
**Dampak:** Tanpa pembanding dan uji signifikansi, klaim "model usulan lebih baik" tidak terbukti.

### Gap 4 — Fitur akustik terlalu sederhana

**Kondisi:** 15 statistik global (mean) dari 45 detik pertama.
**Celah:** State of the art memakai embedding pra-latih: MERT, wav2vec2, OpenL3, CLAP, atau setidaknya statistik temporal (mean + std + delta) dan fitur ritmis yang lebih kaya.
**Peluang:** Bandingkan dua representasi: (a) baseline 15 fitur global, (b) embedding pra-latih. Ukur dampaknya ke Evenness audio dan Smoothness. Ini eksperimen ablation yang murah dan terdengar meyakinkan.

### Gap 5 — Katalog kecil & tidak realistis

**Kondisi:** 1.802 lagu, 45 detik klip, mayoritas satu artis sintetis (`Artist DEAM`), tanpa metadata genre/popularitas.
**Celah:** Rekomendasi nyata butuh katalog puluhan ribu–juta lagu, plus constraint: durasi, eksplisit, ketersediaan, keragaman artis.
**Peluang:** Nyatakan sebagai limitasi, atau tambahkan constraint buatan (maksimal 2 lagu per artis, total durasi, jarak minimum antar lagu) dan ukur trade-off-nya terhadap Smoothness.

### Gap 6 — Tidak ada personalisasi

**Kondisi:** Playlist hanya ditentukan oleh titik awal & tujuan. Dua pengguna dengan pasangan sama mendapat playlist identik.
**Celah:** Sistem rekomendasi nyata menggabungkan preferensi pengguna dengan lintasan emosi.
**Peluang:** Tambahkan profil preferensi (misal vektor selera di ruang audio) sebagai komponen skor Tahap 2: `score = α·jarak_akustik + β·kecocokan_selera`. Ukur dampaknya.

### Gap 7 — Evaluasi hanya satu arah di single run

**Kondisi:** `run_experiment.py` hanya Q3→Q1 dengan satu pasangan lagu (seed 101/202).
**Celah:** Satu sampel tidak bisa menyimpulkan apa pun; tidak ada interval kepercayaan.
**Peluang:** Sudah sebagian dijawab batch (540), tapi tetap: laporkan mean ± std, bukan satu angka; lakukan uji signifikansi antar metrik.

### Gap 8 — Leakage metodologis yang harus dinyatakan

**Kondisi:** PCA/Yeo-Johnson/MinMax di-`fit` pada seluruh 1.802 lagu, bukan train.
**Celah:** Untuk retrieval ini dapat dibenarkan (katalog = database, bukan data latih), tetapi akan ditanyakan.
**Peluang:** Tulis eksplisit di Bab Metodologi + tunjukkan `transform()` sudah tersedia. Jika nanti membangun model learned, wajib pakai split yang sudah ada (1.261/270/271).

---

## 8. Posisi Baseline di TA: Klaim Aman vs Klaim Berisiko

| Klaim | Status | Alasan |
| :--- | :---: | :--- |
| "Mereplikasi baseline Gaur (2023) pada DEAM" | ✅ Aman | Sudah dijalankan, angka terdokumentasi |
| "Pipeline dua tahap: KD-Tree emosi + re-ranking akustik" | ✅ Aman | Persis isi kode |
| "KNN dipakai sebagai metode retrieval, bukan model terlatih" | ✅ Aman | Terverifikasi (nol optimizer/loss) |
| "PCA 12D menjelaskan 99,10% variance fitur akustik" | ✅ Aman | Terukur, Mode B |
| "Smoothness 0,995 pada rute diagonal" | ✅ Aman | Terukur, dengan catatan §6.2 |
| "Baseline adalah model AI" | ⚠️ Berisiko | Tidak ada model yang dilatih |
| "Baseline mencapai akurasi 99%" | ❌ Salah | Itu explained variance PCA |
| "Tiga metrik diuji pada 540 playlist" | ❌ Salah | Batch hanya cosine & euclidean |
| "Fitur IQR di-discretize ke 20 bin" | ❌ Tidak terjadi | Hanya clipping (§6.1) |
| "Model dilatih dengan wandb (CNN, CIFAR-100)" | ❌ Salah | Template palsu (§6.12) |
| "Split 70/15/15 dipakai untuk evaluasi" | ❌ Tidak dipakai | Manifest ada, tidak dipakai skrip |

---

## 9. Cara Membaca Artefak

| File | Untuk apa |
| :--- | :--- |
| `experiments/results/metrics_comparison_deam.csv` | Tabel metrik single run (3 metrik) → Bab Hasil tabel 1 |
| `experiments/results/baseline_cosine_deam_playlist.csv` | Playlist 12 lagu, kolom `step`, `track_id`, V, A, `pca_*` |
| `experiments/results/baseline_deam_traversal_plot.png` | Gambar lintasan Russell Circumplex → Bab Hasil gambar 1 |
| `experiments/results/batch_quadrant_experiments_deam.csv` | 540 baris mentah → analisis K/N/rute |
| `experiments/results/batch_quadrant_summary_deam.csv` | Agregat 12 baris (rute × metrik) → tabel ringkas |
| `experiments/splits/*.csv`, `split_summary.json` | Bukti anti-leakage untuk model usulan (belum dipakai baseline) |
| `data/processed/deam_extracted_features.csv` | Fitur akustik nyata; regenerate bila hilang |

---

## 10. Rekomendasi Langkah Berikutnya (Prioritas)

1. **Bersihkan jebakan klaim** (30 menit): hapus/ganti `wandb-set-up.py` + folder `wandb/`; perbaiki klaim "KBinsDiscretizer" di README/PANDUAN; pastikan tidak ada dokumen yang menyebut "training" untuk baseline.
2. **Tulis Bab Hasil baseline** (1–2 jam): pakai tabel §5.1–5.2, sertakan pembanding Mode A vs Mode B, nyatakan limitasi §6.
3. **Bangun baseline trivial** (2–3 jam): random walk, sorted-by-valence, nearest-neighbor tanpa waypoint. Ini yang membuat angka 0.99 punya makna.
4. **Uji signifikansi** (1 jam): Wilcoxon signed-rank pada 540 baris yang sudah paired per trial.
5. **Bangun model usulan** (inti TA, hitungan hari): manfaatkan anotasi 2 Hz. Mulai dari yang paling murah:
   - rekonstruksi kurva V/A target (sequence-to-sequence),
   - atau planner yang memilih waypoint adaptif (bukan garis lurus),
   - loss: Smoothness + Evenness + endpoint + deviasi dari kurva dinamis.
6. **Ablation** (setelah model jalan): fitur global vs embedding pra-latih; K kecil vs besar; dengan/tanpa personalisasi.

---

## 11. Lampiran

### 11.1 Perintah penting

```fish
cd ~/TA-123140205/Code/Baseline-KNN-Gaur
source .venv/bin/activate.fish

.venv/bin/python tests/test_baseline.py
.venv/bin/python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7
.venv/bin/python experiments/run_quadrant_experiments.py

# regenerasi fitur akustik (bila CSV hilang) — ±35 menit sequential
.venv/bin/python src/deam_extractor.py
```

### 11.2 Glosarium

| Istilah | Arti di repo ini |
| :--- | :--- |
| **V / A** | Valence (negatif–positif) dan Arousal (tenang–energik), skala [-1, 1] |
| **Waypoint (S_T)** | Titik target antara yang dihitung Persamaan 5 |
| **Tahap 1** | Pencarian K kandidat berdasarkan kedekatan emosi di KD-Tree |
| **Tahap 2** | Pemilihan 1 dari K kandidat berdasarkan kedekatan fitur akustik |
| **Smoothness** | \|PCC(V, A)\| playlist; makin 1 makin lurus lintasan |
| **Evenness** | Varians ukuran langkah; makin 0 makin seragam |
| **Mode A** | Fitur akustik sintetis deterministik (`random_state=42`) |
| **Mode B** | Fitur akustik nyata dari 1.802 MP3 via librosa |
| **Lazy learning** | Kelas algoritma tanpa fase training (KNN) |
| **Leakage** | Parameter transformasi di-fit pada data yang seharusnya hanya untuk evaluasi |

### 11.3 Angka acuan cepat

```text
Dataset              : 1.802 track, 45 detik, anotasi statis + dinamis 2 Hz
Distribusi kuadran   : Q3 742 (41.18%) | Q1 594 (32.96%) | Q4 237 (13.15%) | Q2 229 (12.71%)
Split                : 1.261 / 270 / 271 (irisan 0)
Fitur akustik        : 15 kolom, 0 NaN, 1.802 song_id unik
PCA                  : 12 komponen, variance 99.10%, rentang [-1, 1]
KD-Tree              : 1.113 titik koordinat unik
Single run (N=12,K=7): PCC 0.995043 / 0.992504 / 0.983973
Batch                : 540 baris, mean PCC 0.8474, median 0.9515
GPU                  : RTX 3070 Ti, CUDA 12.8, VRAM terpakai 0.00 MB
```
