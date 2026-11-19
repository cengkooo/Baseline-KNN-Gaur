"""
Evaluation Metrics for Playlist Quality
Paper / Thesis: Shaurya Gaur (2023), "Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"

Metrics Implemented:
1. Smoothness (Emotional Path Adherence):
   Pearson Correlation Coefficient (PCC) between Valence and Arousal trajectory points:
       rho = | Cov(V, A) / (sigma_V * sigma_A) |
   Higher PCC indicates higher emotional linearity along the diagonal path.
   
2. Evenness (Step Size Consistency across Transitions):
   Root Mean Square Error of consecutive step sizes normalized by sqrt(D):
       RMSE_k = || S_{k+1} - S_k ||_2 / sqrt(D)
   Evenness is measured by the Variance of RMSE:
       Var(RMSE) = (1 / (N - 1)) * sum( (RMSE_k - mean_RMSE)^2 )
   Lower variance indicates more uniform transition step sizes between songs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr


def calculate_smoothness_pcc(playlist_df: pd.DataFrame, val_col: str = 'valence', aro_col: str = 'arousal') -> float:
    """
    Computes Smoothness via Pearson Correlation Coefficient (PCC) between Valence and Arousal.
    Returns the absolute correlation |rho| in [0, 1].
    
    If standard deviation of either axis is zero (e.g. perfectly horizontal/vertical trajectory),
    returns 1.0 (indicating perfect linear step consistency along a single axis).
    """
    v = playlist_df[val_col].values.astype(np.float64)
    a = playlist_df[aro_col].values.astype(np.float64)
    
    if len(v) < 3:
        return 1.0
        
    std_v = np.std(v)
    std_a = np.std(a)
    
    if std_v < 1e-6 or std_a < 1e-6:
        return 1.0
        
    r, _ = pearsonr(v, a)
    if np.isnan(r):
        return 0.0
    return float(np.abs(r))


def calculate_evenness_variance(playlist_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[float, float]:
    """
    Computes Evenness via Normalized Step-Size RMSE Variance (Equation 6 & 7 in Gaur 2023).
    
    Formula:
        RMSE_k = || S_{k+1} - S_k ||_2 / sqrt(D)
        evenness_variance = Var(RMSE)
        
    Args:
        playlist_df: DataFrame of ordered songs in the playlist.
        feature_cols: List of features to consider (e.g. 2D ['valence', 'arousal'] or 12D audio features).
        
    Returns:
        Tuple of (evenness_variance, mean_rmse).
    """
    matrix = playlist_df[feature_cols].values.astype(np.float64)
    n_songs = len(matrix)
    d = len(feature_cols)
    
    if n_songs < 2:
        return 0.0, 0.0
        
    # Calculate consecutive L2 differences
    diffs = matrix[1:] - matrix[:-1]
    l2_norms = np.linalg.norm(diffs, axis=1)
    
    # Normalize by sqrt(D)
    step_rmse = l2_norms / np.sqrt(d)
    
    evenness_var = float(np.var(step_rmse, ddof=1)) if len(step_rmse) > 1 else 0.0
    mean_rmse = float(np.mean(step_rmse))
    
    return evenness_var, mean_rmse


def evaluate_playlist(playlist_df: pd.DataFrame, 
                      audio_feature_cols: List[str], 
                      va_cols: List[str] = ['valence', 'arousal']) -> Dict[str, float]:
    """
    Aggregates all evaluation metrics for a single generated playlist.
    
    Returns:
        Dictionary containing:
        - 'smoothness_va_pcc': Smoothness in emotion space
        - 'evenness_va_var': Evenness variance in emotion space (lower is better)
        - 'evenness_va_mean_rmse': Average emotional step size
        - 'evenness_audio_var': Evenness variance in 12D audio space
        - 'evenness_audio_mean_rmse': Average audio feature transition distance
    """
    smoothness = calculate_smoothness_pcc(playlist_df, val_col=va_cols[0], aro_col=va_cols[1])
    va_var, va_mean = calculate_evenness_variance(playlist_df, va_cols)
    audio_var, audio_mean = calculate_evenness_variance(playlist_df, audio_feature_cols)
    
    return {
        'smoothness_va_pcc': smoothness,
        'evenness_va_var': va_var,
        'evenness_va_mean_rmse': va_mean,
        'evenness_audio_var': audio_var,
        'evenness_audio_mean_rmse': audio_mean
    }
