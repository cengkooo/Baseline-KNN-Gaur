"""
Batch Quadrant Experiments on DEAM Dataset (Shaurya Gaur 2023 Baseline Replication).
Tests 6 quadrant transition routes:
  1. Horizontal: Q1 <-> Q2, Q3 <-> Q4
  2. Vertical:   Q1 <-> Q4, Q2 <-> Q3
  3. Diagonal:   Q1 <-> Q3, Q2 <-> Q4
Evaluates across variations of K neighbors and playlist length N.
Outputs comprehensive summary statistics to experiments/results/
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable or []
            self.total = total or (len(self.iterable) if hasattr(self.iterable, '__len__') else 0)
            self.n = 0
        def __iter__(self):
            return iter(self.iterable)
        def update(self, n=1):
            self.n += n
        def close(self):
            pass

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import GaurPreprocessor
from src.dataset import SongDatasetClass
from src.deam_loader import load_or_generate_deam_dataset, get_default_deam_path
from src.playlist_generator import TwoStagePlaylistGenerator
from src.metrics import evaluate_playlist
from src.utils_cuda import get_torch_device, log_vram_usage


def run_batch_quadrant_experiments(n_trials: int = 10, output_csv: str = None):
    print("=" * 75)
    print(" BATCH QUADRANT EXPERIMENTS ON DEAM DATASET (GAUR 2023 BASELINE)")
    print("=" * 75)
    
    device = get_torch_device(prefer_cuda=True)
    log_vram_usage()

    deam_dir = get_default_deam_path()
    df_raw, audio_cols = load_or_generate_deam_dataset(deam_root=deam_dir, random_state=42)
    print(f"[+] Loaded DEAM: {len(df_raw)} songs, {len(audio_cols)} audio features.")

    preprocessor = GaurPreprocessor(n_pca_components=12, iqr_multiplier=5.5, n_bins=20)
    df_clean, pca_cols = preprocessor.fit_transform(df_raw, audio_feature_cols=audio_cols)

    dataset = SongDatasetClass(df=df_clean, audio_feature_cols=pca_cols)
    print(f"[+] KD-Tree indexed on {len(dataset.unique_coords)} unique DEAM points.")

    routes = [
        ("Diagonal_Q3_to_Q1", 3, 1),
        ("Diagonal_Q2_to_Q4", 2, 4),
        ("Horizontal_Q1_to_Q2", 1, 2),
        ("Horizontal_Q3_to_Q4", 3, 4),
        ("Vertical_Q1_to_Q4", 1, 4),
        ("Vertical_Q2_to_Q3", 2, 3),
    ]

    k_options = [5, 7, 15]
    n_options = [5, 12, 19]
    metrics_options = ['cosine', 'euclidean']

    records = []
    total_combinations = len(routes) * len(k_options) * len(n_options) * len(metrics_options) * n_trials
    print(f"[*] Running {total_combinations} playlist evaluations...")

    pbar = tqdm(total=total_combinations, desc="Evaluating Playlists")

    for route_name, q_start, q_dest in routes:
        for k in k_options:
            for length in n_options:
                for metric in metrics_options:
                    generator = TwoStagePlaylistGenerator(dataset=dataset, k_neighbors=k, metric=metric)
                    for trial_idx in range(n_trials):
                        seed = 1000 + trial_idx * 17
                        try:
                            s_start = dataset.get_song_in_quadrant(q_start, seed=seed)
                            s_dest = dataset.get_song_in_quadrant(q_dest, seed=seed + 999)
                            
                            p_df = generator.generate_playlist(
                                start_track_id=s_start['track_id'],
                                dest_track_id=s_dest['track_id'],
                                playlist_length=length
                            )
                            eval_res = evaluate_playlist(p_df, audio_feature_cols=pca_cols)
                            
                            records.append({
                                'Route': route_name,
                                'Start_Quadrant': q_start,
                                'Dest_Quadrant': q_dest,
                                'K_Neighbors': k,
                                'Playlist_Length': length,
                                'Distance_Metric': metric,
                                'Trial': trial_idx,
                                'Smoothness_PCC': eval_res['smoothness_va_pcc'],
                                'Evenness_VA_Var': eval_res['evenness_va_var'],
                                'Evenness_Audio_Var': eval_res['evenness_audio_var'],
                                'Mean_VA_Step': eval_res['evenness_va_mean_rmse'],
                                'Mean_Audio_Step': eval_res['evenness_audio_mean_rmse']
                            })
                        except Exception as e:
                            pass
                        pbar.update(1)

    pbar.close()

    results_df = pd.DataFrame(records)
    if output_csv is None:
        out_dir = os.path.join(current_dir, "results")
        os.makedirs(out_dir, exist_ok=True)
        output_csv = os.path.join(out_dir, "batch_quadrant_experiments_deam.csv")

    results_df.to_csv(output_csv, index=False)
    print(f"\n[+] Successfully saved {len(results_df)} batch experiment rows to:\n    {output_csv}")

    # Summary table
    grouped = results_df.groupby(['Route', 'Distance_Metric'])[['Smoothness_PCC', 'Evenness_VA_Var', 'Evenness_Audio_Var']].mean().reset_index()
    print("\n=== AGGREGATED BENCHMARK BY ROUTE & METRIC ===")
    print(grouped.to_string(index=False))


if __name__ == '__main__':
    run_batch_quadrant_experiments(n_trials=5)
