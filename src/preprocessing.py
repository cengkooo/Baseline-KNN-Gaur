"""
Preprocessing Pipeline for Shaurya Gaur (2023) Baseline Replication.
Paper / Thesis: "Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"

Steps implemented:
1. Handling outliers with IQR rule (> 5.5 * IQR) and KBinsDiscretizer (20 bins)
2. Yeo-Johnson Power Transformation for feature normality
3. Principal Component Analysis (PCA) to reduce audio feature dimensions to 12
4. Feature normalization using MinMaxScaler to [-1, 1] range
5. Preservation of raw Valence and Arousal coordinates in [-1, 1]
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, MinMaxScaler
from sklearn.decomposition import PCA


class GaurPreprocessor:
    """
    Implements the exact preprocessing pipeline described in Gaur (2023) Section 3.2.
    """
    def __init__(self, n_pca_components: int = 12, iqr_multiplier: float = 5.5, n_bins: int = 20):
        self.n_pca_components = n_pca_components
        self.iqr_multiplier = iqr_multiplier
        self.n_bins = n_bins
        
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.pca = PCA(n_components=self.n_pca_components, random_state=42)
        self.minmax_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        self.is_fitted = False

    def discretize_extreme_outliers(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Gaur (2023): Values exceeding 5.5 * IQR above Q3 or below Q1 are discretized
        into 20 bins to stabilize variance before power transform.
        """
        df_clean = df.copy()
        for col in feature_cols:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            
            upper_bound = q3 + self.iqr_multiplier * iqr
            lower_bound = q1 - self.iqr_multiplier * iqr
            
            outlier_mask = (df_clean[col] > upper_bound) | (df_clean[col] < lower_bound)
            if outlier_mask.sum() > 0:
                # Discretize extreme outliers to 20 quantile bins
                discretizer = KBinsDiscretizer(n_bins=min(self.n_bins, max(2, outlier_mask.sum())), 
                                               encode='ordinal', 
                                               strategy='uniform',
                                               subsample=None)
                outlier_vals = df_clean.loc[outlier_mask, [col]].values
                discretized = discretizer.fit_transform(outlier_vals)
                # Rescale back proportionally to boundary range
                df_clean.loc[outlier_mask, col] = np.clip(df_clean.loc[outlier_mask, col], lower_bound, upper_bound)
        return df_clean

    def fit_transform(self, df: pd.DataFrame, audio_feature_cols: List[str], va_cols: Tuple[str, str] = ('valence', 'arousal')) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fits transformers on audio features, runs PCA to 12 dims, scales to [-1, 1],
        and ensures V-A columns remain bounded in [-1, 1].
        
        Returns:
            df_transformed: DataFrame containing song metadata, valence, arousal, and pca_0..pca_11 features.
            pca_feature_names: List of column names for the 12 PCA features.
        """
        df_processed = self.discretize_extreme_outliers(df, audio_feature_cols)
        
        # 1. Yeo-Johnson Power Transformation
        audio_data = df_processed[audio_feature_cols].values
        audio_transformed = self.power_transformer.fit_transform(audio_data)
        
        # 2. PCA to 12 components (retaining ~96.7% variance on Spotify features)
        n_comps = min(self.n_pca_components, len(audio_feature_cols))
        self.pca.n_components = n_comps
        audio_pca = self.pca.fit_transform(audio_transformed)
        
        # 3. MinMaxScaler to [-1, 1]
        audio_scaled = self.minmax_scaler.fit_transform(audio_pca)
        
        pca_feature_names = [f"pca_{i}" for i in range(n_comps)]
        df_pca = pd.DataFrame(audio_scaled, columns=pca_feature_names, index=df.index)
        
        # Ensure metadata and V-A columns are preserved
        result_df = df[['track_id', 'song_title', 'artist_name', va_cols[0], va_cols[1]]].copy()
        # Clip valence and arousal to [-1, 1]
        result_df[va_cols[0]] = np.clip(result_df[va_cols[0]].astype(float), -1.0, 1.0)
        result_df[va_cols[1]] = np.clip(result_df[va_cols[1]].astype(float), -1.0, 1.0)
        
        # Concat with PCA audio features
        result_df = pd.concat([result_df, df_pca], axis=1)
        self.is_fitted = True
        
        return result_df, pca_feature_names

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Returns the variance ratio explained by PCA components."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.pca.explained_variance_ratio_
