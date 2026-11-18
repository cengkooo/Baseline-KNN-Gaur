"""
DEAM Dataset Splitting Utilities for Playlist Generation & MER.
Academic Standards:
1. Stratified Partitioning based on Russell's 4 Emotion Quadrants to address class imbalance.
2. Anti-Data Leakage: Parameter fitting strictly on Training set.
3. Dual-Scenario Support:
   - Scenario A: Supervised Learning (Train 70% / Val 15% / Test 15%)
   - Scenario B: Catalog Retrieval (Music Search Pool 80% / Unseen Query Seeds 20%)
   - Scenario C: Stratified 5-Fold Cross Validation
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Generator
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def assign_emotion_quadrant(df: pd.DataFrame, 
                            val_col: str = 'valence', 
                            aro_col: str = 'arousal') -> pd.Series:
    """
    Assigns each track to one of Russell's Circumplex 4 Quadrants:
    - Q1: Happy / Excited (Valence > 0, Arousal > 0)
    - Q2: Angry / Tense   (Valence <= 0, Arousal > 0)
    - Q3: Sad / Depressed (Valence <= 0, Arousal <= 0)
    - Q4: Calm / Relaxed  (Valence > 0, Arousal <= 0)
    """
    v = df[val_col].values
    a = df[aro_col].values
    
    conditions = [
        (v > 0) & (a > 0),
        (v <= 0) & (a > 0),
        (v <= 0) & (a <= 0),
        (v > 0) & (a <= 0)
    ]
    choices = ['Q1_Happy', 'Q2_Angry', 'Q3_Sad', 'Q4_Calm']
    return pd.Series(np.select(conditions, choices, default='Q3_Sad'), index=df.index, name='quadrant')


def get_quadrant_distribution(df: pd.DataFrame, quadrant_col: str = 'quadrant') -> Dict[str, dict]:
    """Returns count and percentage per quadrant."""
    counts = df[quadrant_col].value_counts()
    total = len(df)
    return {
        q: {
            "count": int(cnt),
            "percentage": round(float(cnt / total * 100), 2)
        }
        for q, cnt in counts.items()
    }


def stratified_split_deam(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 70/15/15 split on DEAM dataset based on 4 emotion quadrants.
    Ensures sparse quadrants (Q2 Angry and Q4 Calm) are equally represented in train, val, and test.
    
    Returns:
        (df_train, df_val, df_test)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    df_copy = df.copy()
    if 'quadrant' not in df_copy.columns:
        df_copy['quadrant'] = assign_emotion_quadrant(df_copy)
        
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(df_copy, df_copy['quadrant']))
    
    df_train = df_copy.iloc[train_idx].copy().reset_index(drop=True)
    df_temp = df_copy.iloc[temp_idx].copy().reset_index(drop=True)
    
    # Second split: Val vs Test
    val_relative_ratio = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_relative_ratio, random_state=random_state)
    val_idx, test_idx = next(sss2.split(df_temp, df_temp['quadrant']))
    
    df_val = df_temp.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].copy().reset_index(drop=True)
    
    return df_train, df_val, df_test


def create_catalog_query_split(
    df: pd.DataFrame,
    catalog_ratio: float = 0.80,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into:
    1. Search Catalog Pool (e.g. 80%): Songs available for the KD-Tree playlist interpolation.
    2. Unseen Query Seeds (e.g. 20%): Tracks used exclusively as start (S_0) and end (S_N) waypoints
       to strictly test retrieval generalization without self-repetition.
    """
    df_copy = df.copy()
    if 'quadrant' not in df_copy.columns:
        df_copy['quadrant'] = assign_emotion_quadrant(df_copy)
        
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - catalog_ratio, random_state=random_state)
    cat_idx, query_idx = next(sss.split(df_copy, df_copy['quadrant']))
    
    df_catalog = df_copy.iloc[cat_idx].copy().reset_index(drop=True)
    df_queries = df_copy.iloc[query_idx].copy().reset_index(drop=True)
    
    return df_catalog, df_queries


def stratified_kfold_deam(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yields (train_fold, test_fold) for stratified 5-fold cross-validation.
    """
    df_copy = df.copy()
    if 'quadrant' not in df_copy.columns:
        df_copy['quadrant'] = assign_emotion_quadrant(df_copy)
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(df_copy, df_copy['quadrant']):
        yield df_copy.iloc[train_idx].copy().reset_index(drop=True), df_copy.iloc[test_idx].copy().reset_index(drop=True)


def save_split_manifests(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """
    Saves split track manifests (CSV and JSON) for reproducible experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "deam_train_manifest.csv")
    val_path = os.path.join(output_dir, "deam_val_manifest.csv")
    test_path = os.path.join(output_dir, "deam_test_manifest.csv")
    summary_path = os.path.join(output_dir, "split_summary.json")
    
    keep_cols = [c for c in ['track_id', 'song_id', 'valence', 'arousal', 'quadrant'] if c in df_train.columns]
    df_train[keep_cols].to_csv(train_path, index=False)
    df_val[keep_cols].to_csv(val_path, index=False)
    df_test[keep_cols].to_csv(test_path, index=False)
    
    summary = {
        "total_tracks": len(df_train) + len(df_val) + len(df_test),
        "train_set": {
            "total": len(df_train),
            "distribution": get_quadrant_distribution(df_train)
        },
        "val_set": {
            "total": len(df_val),
            "distribution": get_quadrant_distribution(df_val)
        },
        "test_set": {
            "total": len(df_test),
            "distribution": get_quadrant_distribution(df_test)
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    return {
        "train_csv": train_path,
        "val_csv": val_path,
        "test_csv": test_path,
        "summary_json": summary_path
    }


if __name__ == "__main__":
    from .deam_loader import load_or_generate_deam_dataset
    print("Testing DEAM Stratified Split...")
    df, _ = load_or_generate_deam_dataset()
    df['quadrant'] = assign_emotion_quadrant(df)
    
    print("\n[Original DEAM Quadrant Distribution]:")
    for q, d in get_quadrant_distribution(df).items():
        print(f"  {q}: {d['count']} songs ({d['percentage']}%)")
        
    tr, va, te = stratified_split_deam(df, 0.70, 0.15, 0.15)
    print(f"\n[Split Sizes]: Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    print("[Train Quadrant Distribution]:")
    for q, d in get_quadrant_distribution(tr).items():
        print(f"  {q}: {d['count']} ({d['percentage']}%)")
