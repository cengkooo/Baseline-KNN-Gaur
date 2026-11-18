"""
End-to-End Experiment Execution for Shaurya Gaur (2023) Baseline.
Primary Dataset: DEAM (Database for Emotion Analysis in Music)
Hardware: PyTorch CUDA (NVIDIA GeForce RTX 3070 Ti) with CPU fallback.

Runs:
1. Loads primary dataset DEAM (1,802 songs) from EXPERIMENT/Dataset/DEAM/ with normalized Valence-Arousal [-1, 1]
2. Preprocessing: IQR Discretizer -> Yeo-Johnson PowerTransformer -> 12D PCA -> MinMaxScaler [-1, 1]
3. SongDatasetClass & KD-Tree indexing on 2D emotion space
4. 4-Quadrant Traversal experiment (e.g. Sad (Q3) -> Happy (Q1))
5. Stage 2 Distance Metric Evaluation (Cosine vs. Euclidean vs. Manhattan)
6. Metrics: Smoothness (PCC) and Evenness (Var(RMSE) / sqrt(D))
7. Trajectory visualization plot saved to experiments/results/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local src directory is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import GaurPreprocessor
from src.dataset import SongDatasetClass, generate_synthetic_dataset
from src.playlist_generator import TwoStagePlaylistGenerator
from src.metrics import evaluate_playlist
from src.deam_loader import load_or_generate_deam_dataset, get_default_deam_path
from src.utils_cuda import get_torch_device, log_vram_usage


def plot_playlist_trajectory(playlist_dfs: dict, save_path: str, dataset_label: str = "DEAM"):
    """Visualizes the 2D emotional path on the Russell Circumplex Model."""
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    plt.text(0.5, 0.85, 'Q1: Happy / Exuberant', fontsize=11, color='green', ha='center', weight='bold')
    plt.text(-0.5, 0.85, 'Q2: Angry / Anxious', fontsize=11, color='red', ha='center', weight='bold')
    plt.text(-0.5, -0.85, 'Q3: Sad / Depressed', fontsize=11, color='blue', ha='center', weight='bold')
    plt.text(0.5, -0.85, 'Q4: Calm / Relaxed', fontsize=11, color='teal', ha='center', weight='bold')
    
    markers = ['o-', 's--', '^:']
    for idx, (label, df) in enumerate(playlist_dfs.items()):
        marker = markers[idx % len(markers)]
        plt.plot(df['valence'], df['arousal'], marker, label=label, linewidth=2, markersize=7)
        # Highlight start and end
        plt.scatter(df.iloc[0]['valence'], df.iloc[0]['arousal'], color='black', s=120, zorder=5)
        plt.scatter(df.iloc[-1]['valence'], df.iloc[-1]['arousal'], color='gold', edgecolor='black', s=150, zorder=5)
        
    plt.title(f"Baseline Gaur (2023): Mood-Dynamic Playlist on {dataset_label}\n2D Emotion Space Traversal", fontsize=12, weight='bold')
    plt.xlabel("Valence (Negative -> Positive)", fontsize=11)
    plt.ylabel("Arousal (Calm -> Energetic)", fontsize=11)
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Trajectory plot successfully saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Shaurya Gaur (2023) Baseline Replication with DEAM Dataset")
    parser.add_argument('--deam_path', type=str, default=None, help="Path to DEAM dataset directory")
    parser.add_argument('--use_deam', action='store_true', default=True, help="Use primary DEAM dataset (default: True)")
    parser.add_argument('--k_neighbors', type=int, default=7, help="K candidates in Stage 1 KNN (Gaur default: 7)")
    parser.add_argument('--playlist_length', type=int, default=12, help="Length N of generated playlist (Gaur default: 12)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for results and plots")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print(" BASELINE REPLICATION: SHAURYA GAUR (2023 / EvoMUSART 2024)")
    print(" Mood-Dynamic Playlist Interpolating Emotions Using Two-Stage KNN")
    print(" Primary Dataset: DEAM (Database for Emotion Analysis in Music)")
    print("=" * 75)

    # 0. Hardware / CUDA status check
    device = get_torch_device(prefer_cuda=True)
    log_vram_usage()

    # 1. Load Dataset (DEAM as Primary Dataset)
    deam_dir = args.deam_path or get_default_deam_path()
    if args.use_deam and os.path.exists(deam_dir):
        print(f"[*] Loading Primary Dataset: DEAM from {deam_dir}...")
        df_raw, audio_cols = load_or_generate_deam_dataset(deam_root=deam_dir, random_state=42)
        dataset_name = "DEAM (1,802 Tracks)"
    else:
        print(f"[*] Generating synthetic dataset (1,500 tracks)...")
        df_raw, audio_cols = generate_synthetic_dataset(n_samples=1500, random_state=42)
        dataset_name = "Synthetic Baseline"

    print(f"[+] Active Dataset: {dataset_name} | Total Tracks: {len(df_raw)} | Audio Features: {len(audio_cols)}")

    # 2. Preprocessing Pipeline
    print("[*] Running Gaur Preprocessing (IQR Discretizer -> Yeo-Johnson -> 12D PCA -> MinMaxScaler [-1, 1])...")
    preprocessor = GaurPreprocessor(n_pca_components=12, iqr_multiplier=5.5, n_bins=20)
    df_clean, pca_cols = preprocessor.fit_transform(df_raw, audio_feature_cols=audio_cols)
    var_exp = preprocessor.get_explained_variance_ratio()
    print(f"[+] PCA 12 Components Fitted. Cumulative Explained Variance: {np.sum(var_exp)*100:.2f}%")

    # 3. Build Spatial Index & Dataset Container
    print("[*] Building KD-Tree over unique Valence-Arousal coordinates...")
    dataset = SongDatasetClass(df=df_clean, audio_feature_cols=pca_cols)
    print(f"[+] KD-Tree indexed over {len(dataset.unique_coords)} unique coordinate points.")

    # 4. Define 4-Quadrant Traversal Test
    # Q3 (Sad) -> Q1 (Happy): Valence [- -> +], Arousal [- -> +]
    song_start = dataset.get_song_in_quadrant(quadrant=3, seed=101)
    song_dest = dataset.get_song_in_quadrant(quadrant=1, seed=202)

    print("\n" + "-" * 75)
    print(f"[*] Benchmark Traversal on {dataset_name}:")
    print(f"    Origin (Q3 Sad)  : '{song_start['song_title']}' [V={song_start['valence']:.3f}, A={song_start['arousal']:.3f}]")
    print(f"    Target (Q1 Happy): '{song_dest['song_title']}' [V={song_dest['valence']:.3f}, A={song_dest['arousal']:.3f}]")
    print(f"    Parameters       : Length N={args.playlist_length}, Stage 1 K={args.k_neighbors}")
    print("-" * 75)

    # 5. Compare Stage 2 Distance Metrics (Cosine vs. Euclidean vs. Manhattan)
    metrics_to_test = ['cosine', 'euclidean', 'manhattan']
    results_summary = []
    playlist_dict = {}

    for metric in metrics_to_test:
        generator = TwoStagePlaylistGenerator(dataset=dataset, k_neighbors=args.k_neighbors, metric=metric)
        playlist_df = generator.generate_playlist(
            start_track_id=song_start['track_id'],
            dest_track_id=song_dest['track_id'],
            playlist_length=args.playlist_length
        )
        playlist_dict[f"Stage 2: {metric.capitalize()}"] = playlist_df
        
        # Calculate Gaur's Evaluation Metrics
        eval_metrics = evaluate_playlist(playlist_df, audio_feature_cols=pca_cols)
        results_summary.append({
            'Metric': metric.capitalize(),
            'Smoothness (PCC)': eval_metrics['smoothness_va_pcc'],
            'Evenness V-A Var': eval_metrics['evenness_va_var'],
            'Evenness Audio Var': eval_metrics['evenness_audio_var'],
            'Mean V-A Step': eval_metrics['evenness_va_mean_rmse'],
            'Mean Audio Step': eval_metrics['evenness_audio_mean_rmse']
        })

    summary_df = pd.DataFrame(results_summary)
    print("\n=== EXPERIMENTAL RESULTS SUMMARY ON DEAM (Gaur 2023 Metrics) ===")
    print(summary_df.to_string(index=False))

    # Save CSV outputs
    summary_path = os.path.join(out_dir, "metrics_comparison_deam.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[+] Saved metrics comparison table to: {summary_path}")

    # Save detailed playlist for winning metric (Cosine)
    winning_df = playlist_dict["Stage 2: Cosine"]
    winning_path = os.path.join(out_dir, "baseline_cosine_deam_playlist.csv")
    winning_df.to_csv(winning_path, index=False)
    print(f"[+] Saved baseline Cosine playlist (length {len(winning_df)}) to: {winning_path}")

    # 6. Plot and save trajectory visualization
    plot_path = os.path.join(out_dir, "baseline_deam_traversal_plot.png")
    plot_playlist_trajectory(playlist_dict, plot_path, dataset_label="DEAM Dataset")

    print("\n[V] DEAM Baseline experiment execution completed successfully!")
    print("=" * 75)


if __name__ == '__main__':
    main()
