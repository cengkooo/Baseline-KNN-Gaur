# Baseline Model Replication: Shaurya Gaur (2023 / EvoMUSART 2024)
## 🎯 Primary Dataset: DEAM (Database for Emotion Analysis in Music)
### ⚡ PyTorch CUDA (NVIDIA GeForce RTX 3070 Ti) & Environment `uv`
#### https://drive.google.com/drive/folders/1YIyq1PhKwOklye8QLa1nu56IjKw8oLUn

> **Paper Title:** *Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm*  
> **Author:** Shaurya Gaur (Master's Thesis, University of Victoria / Springer EvoMUSART 2024)  
> **Primary Dataset:** **DEAM (`EXPERIMENT/Dataset/DEAM/`)** — 1,802 audio tracks (`.mp3`, 45s) with static & dynamic (2 Hz) emotion annotations.  
> **Hardware Target:** NVIDIA GeForce RTX 3070 Ti (8GB GDDR6X, CUDA Compute 8.6) + PyTorch 2.x  
> **Package / Environment Manager:** `uv` (Astral)  
> **Academic Standard:** Open Data ITERA Bab 8 & Code Walkthrough Ready.

---

## 📁 Repository Structure

```text
EXPERIMENT/Code/baseline_gaur/
├── .gitignore                          # Excludes large audio files, venv, and checkpoints
├── pyproject.toml                      # Standard PEP 621 metadata & uv dependencies
├── requirements.txt                    # Python dependencies (PyTorch CUDA, librosa, etc.)
├── README.md                           # Documentation & reproduction guide
├── LICENSE                             # MIT Open Source License
├── src/
│   ├── __init__.py                     # Module package exports
│   ├── utils_cuda.py                   # ⚡ Helper for RTX 3070 Ti CUDA device & VRAM tracking
│   ├── deam_loader.py                  # 🎵 Loader for DEAM static & dynamic annotations
│   ├── preprocessing.py               # ⚙️ Pipeline: IQR Discretizer, Yeo-Johnson, 12D PCA, MinMax [-1, 1]
│   ├── dataset.py                      # 📦 SongDatasetClass (KD-Tree on V-A space) + fallback generator
│   ├── stage1_knn.py                   # 🎯 Stage 1: Linear Waypoint Interpolation (Eq. 5) & KD-Tree KNN
│   ├── stage2_selector.py              # 🎵 Stage 2: Audio Distance Re-Ranking via PyTorch CUDA tensors
│   ├── playlist_generator.py           # 🔄 TwoStagePlaylistGenerator (Gaur Algorithm 1)
│   ├── split_utils.py                  # ✂️ Stratified 4-Quadrant Train/Val/Test Split & Anti-Data Leakage
│   └── metrics.py                      # 📊 Metrics: Smoothness (PCC) & Evenness (Var(RMSE)/sqrt(D))
└── experiments/
    ├── run_experiment.py               # End-to-end runnable script on DEAM dataset
    ├── run_quadrant_experiments.py      # Batch 540 playlist benchmark script across 6 routes
    ├── configs/
    │   └── params_deam.json            # Grid hyperparameter configurations (K, N, metrics)
    ├── splits/
    │   ├── deam_train_manifest.csv     # 70% Stratified Training set (1,261 tracks)
    │   ├── deam_val_manifest.csv       # 15% Stratified Validation set (270 tracks)
    │   ├── deam_test_manifest.csv      # 15% Stratified Test set (271 tracks)
    │   └── split_summary.json          # Manifest distribution statistics
    └── results/
        ├── metrics_comparison_deam.csv # Comparison table (Cosine vs Euclidean vs Manhattan)
        ├── baseline_cosine_deam_playlist.csv # Generated 12-track playlist traversal
        └── baseline_deam_traversal_plot.png  # 2D Russell emotion space trajectory plot
```

---

## 🔬 Mathematical Formulations

### 1. DEAM Emotion Normalization ($[1.0, 9.0] \to [-1.0, +1.0]$)
$$V_{\text{norm}} = \frac{V_{\text{DEAM}} - 5.0}{4.0}, \quad A_{\text{norm}} = \frac{A_{\text{DEAM}} - 5.0}{4.0}$$

### 2. Stage 1: Linear Waypoint Interpolation (Gaur 2023 Eq. 5)
$$S_T = S_L + \frac{S_N - S_L}{N - L}$$
- $S_T$: Target waypoint in 2D Emotion Space $(V, A)$
- $S_L$: Current last song in playlist at step $L$
- $S_N$: Final destination song at step $N$
- Queries KD-Tree for $K = 7$ nearest candidate tracks.

### 3. Stage 2: Audio Feature Distance Re-Ranking (PyTorch CUDA)
Evaluates 12-dimensional PCA acoustic features:
$$\text{Dist}_{\text{Cosine}}(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

### 4. Official Evaluation Metrics
- **Smoothness (PCC - Eq. 6):**
  $$\rho = \left| \frac{\text{Cov}(V, A)}{\sigma_V \sigma_A} \right|$$
- **Evenness (Step Size Variance - Eq. 7):**
  $$\text{RMSE}_k = \frac{\|S_{k+1} - S_k\|_2}{\sqrt{D}}, \quad \text{Evenness} = \text{Var}(\text{RMSE})$$

---

## ✂️ DEAM Dataset Splitting Methodology (Anti-Data Leakage)

Due to severe class imbalance in DEAM (Sad tracks account for ~41% while Angry and Calm are only ~13% each), random splitting causes severe under-representation of sparse quadrants. We implement **Stratified Sampling based on Russell's 4 Emotion Quadrants**:

1. **Supervised / MER Split (70% Train / 15% Val / 15% Test):**
   - **Train (1,261 tracks):** Fits Yeo-Johnson transformer, 12D PCA projection, and MinMax bounds.
   - **Validation (270 tracks) & Test (271 tracks):** Transformed strictly using parameters learned from Train (`GaurPreprocessor.transform()`), guaranteeing **Zero Data Leakage**.
2. **Catalog Retrieval Split (80% Catalog Pool / 20% Unseen Query Seeds):**
   - **Catalog Pool (80%):** Populates the KD-Tree search space.
   - **Query Seeds (20%):** Serves exclusively as unseen start ($S_0$) and destination ($S_N$) waypoints.
3. **Execution Command:**
   ```powershell
   python -c "from src import load_or_generate_deam_dataset, stratified_split_deam, assign_emotion_quadrant, save_split_manifests; df, _ = load_or_generate_deam_dataset(); df['quadrant'] = assign_emotion_quadrant(df); tr, va, te = stratified_split_deam(df, 0.7, 0.15, 0.15); save_split_manifests(tr, va, te, 'experiments/splits')"
   ```

---

## 🚀 Setup & Execution Guide

### 1. Virtual Environment Setup with `uv`
```powershell
# Navigate to project folder
cd "EXPERIMENT/Code/baseline_gaur"

# Create virtual environment with Python 3.11
uv venv .venv --python 3.11

# Activate virtual environment
.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate  # Linux / macOS
```

### 2. Install PyTorch with CUDA Acceleration
```powershell
# Install PyTorch with CUDA 12.1 for NVIDIA RTX 3070 Ti
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
uv pip install -r requirements.txt
```

### 3. Verify CUDA & GPU Hardware
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Model:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 4. Run Experiment Pipeline on DEAM
```powershell
# Runs end-to-end benchmark on 1,802 DEAM tracks
python experiments/run_experiment.py --playlist_length 12 --k_neighbors 7
```

Outputs are automatically saved to `experiments/results/`:
- `metrics_comparison_deam.csv`
- `baseline_cosine_deam_playlist.csv`
- `baseline_deam_traversal_plot.png`

---

## 🐙 Git & GitHub Workflow

```powershell
# 1. Initialize git
git init
git branch -M main

# 2. Add files (.gitignore automatically excludes raw audio and .venv)
git add .
git commit -m "feat: implement Gaur (2023) baseline on DEAM dataset with PyTorch CUDA and uv"

# 3. Push to remote
git remote add origin https://github.com/your-username/baseline-mood-dynamic-playlist.git
git push -u origin main
```
