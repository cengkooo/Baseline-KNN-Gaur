"""
DEAM Dataset Loader and Normalization Module.
Dataset Path: EXPERIMENT/Dataset/DEAM/
Location of Static Annotations:
  - EXPERIMENT/Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv
  - EXPERIMENT/Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv
Location of Dynamic Annotations (2 Hz, 500ms):
  - EXPERIMENT/Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv
  - EXPERIMENT/Dataset/DEAM/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv
Location of Audio MP3s:
  - EXPERIMENT/Dataset/DEAM/DEAM_audio/MEMD_audio/{song_id}.mp3 (1,802 files)
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def get_default_deam_path() -> str:
    """Finds the absolute path to Dataset/DEAM (repo-relative, symlink-safe)."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_file_dir, ".."))
    candidates = [
        os.path.join(repo_root, "Dataset", "DEAM"),
        os.path.abspath(os.path.join("Dataset", "DEAM")),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_deam_static_annotations(deam_root: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and concatenates DEAM static annotations (1-2000 and 2000-2058).
    Converts valence and arousal from Likert [1, 9] to [-1.0, +1.0] scale:
        V_norm = (valence_mean - 5.0) / 4.0
        A_norm = (arousal_mean - 5.0) / 4.0
    """
    if deam_root is None:
        deam_root = get_default_deam_path()
        
    song_level_dir = os.path.join(
        deam_root, "DEAM_Annotations", "annotations", 
        "annotations averaged per song", "song_level"
    )
    
    csv_1 = os.path.join(song_level_dir, "static_annotations_averaged_songs_1_2000.csv")
    csv_2 = os.path.join(song_level_dir, "static_annotations_averaged_songs_2000_2058.csv")
    
    if not os.path.exists(csv_1):
        raise FileNotFoundError(f"DEAM static annotations not found at: {csv_1}")
        
    df1 = pd.read_csv(csv_1)
    df2 = pd.read_csv(csv_2) if os.path.exists(csv_2) else pd.DataFrame()
    
    df = pd.concat([df1, df2], ignore_index=True)
    # Strip whitespace from columns
    df.columns = [c.strip() for c in df.columns]
    
    # Clean up song_id
    df['song_id'] = df['song_id'].astype(int)
    df = df.drop_duplicates(subset=['song_id']).reset_index(drop=True)
    
    # Scale from [1, 9] to [-1, 1]
    df['valence'] = np.clip((df['valence_mean'] - 5.0) / 4.0, -1.0, 1.0)
    df['arousal'] = np.clip((df['arousal_mean'] - 5.0) / 4.0, -1.0, 1.0)
    
    # Format metadata columns
    df['track_id'] = df['song_id'].apply(lambda x: f"DEAM_{x:04d}")
    df['song_title'] = df['song_id'].apply(lambda x: f"DEAM Clip {x}")
    df['artist_name'] = df['song_id'].apply(lambda x: f"Artist DEAM")
    
    return df


def load_or_generate_deam_dataset(deam_root: Optional[str] = None, 
                                   features_csv_path: Optional[str] = None,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads DEAM static annotations and combines with acoustic audio features.
    If pre-extracted audio features are available at `features_csv_path`, loads them.
    Otherwise, generates simulated acoustic audio features for the 1,802 DEAM tracks
    so that the pipeline can run immediately with the exact real DEAM V-A coordinates.
    """
    df_static = load_deam_static_annotations(deam_root)
    n_tracks = len(df_static)
    
    if features_csv_path is None:
        # Automatically check if pre-extracted features exist in data/processed
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_processed = os.path.abspath(os.path.join(current_file_dir, "..", "data", "processed", "deam_extracted_features.csv"))
        if os.path.exists(candidate_processed):
            features_csv_path = candidate_processed

    if features_csv_path and os.path.exists(features_csv_path):
        df_feat = pd.read_csv(features_csv_path)
        df_feat.columns = [c.strip() for c in df_feat.columns]
        merged = pd.merge(df_static, df_feat, on='song_id', how='inner')
        # The features CSV contains exactly the acoustic feature columns plus song_id.
        # Deriving audio_cols from it (instead of "everything not in meta_cols") avoids
        # picking up annotation columns such as 'valence_ max_mean' from the second
        # static-annotation file.
        audio_cols = [c for c in df_feat.columns if c != 'song_id']
        return merged, audio_cols
        
    # Generate 15 acoustic audio features mimicking MFCC, Spectral, and Energy distributions
    rng = np.random.default_rng(random_state)
    audio_feat_names = [
        "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6",
        "spectral_centroid", "spectral_rolloff", "spectral_flatness",
        "zero_crossing_rate", "rms_energy", "tempo_bpm",
        "chroma_1", "chroma_2", "chroma_3"
    ]
    
    audio_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_tracks, len(audio_feat_names)))
    # Correlate energy with arousal and valence slightly (natural musical correlation)
    audio_matrix[:, 10] += df_static['arousal'].values * 0.5  # rms_energy correlated with arousal
    audio_matrix[:, 6] += df_static['valence'].values * 0.3   # spectral_centroid correlated with valence
    
    for idx, col in enumerate(audio_feat_names):
        df_static[col] = audio_matrix[:, idx]
        
    return df_static, audio_feat_names


def load_deam_dynamic_annotations(deam_root: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads DEAM dynamic time-series annotations (2 Hz sampling, 500 ms intervals).
    Returns (df_valence_dynamic, df_arousal_dynamic).
    Used as the bridge for the proposed dynamic trajectory alignment model.
    """
    if deam_root is None:
        deam_root = get_default_deam_path()
        
    dynamic_dir = os.path.join(
        deam_root, "DEAM_Annotations", "annotations", 
        "annotations averaged per song", "dynamic (per second annotations)"
    )
    val_path = os.path.join(dynamic_dir, "valence.csv")
    aro_path = os.path.join(dynamic_dir, "arousal.csv")
    
    if not os.path.exists(val_path) or not os.path.exists(aro_path):
        raise FileNotFoundError(f"Dynamic annotations not found in: {dynamic_dir}")
        
    df_val = pd.read_csv(val_path)
    df_aro = pd.read_csv(aro_path)
    
    return df_val, df_aro
