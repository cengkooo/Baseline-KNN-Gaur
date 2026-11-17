"""
Song Dataset Class & KD-Tree spatial index for Gaur (2023) Baseline Replication.
Paper / Thesis: "Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"

Gaur's architecture separates spatial V-A coordinates from song instances:
Multiple songs can share the same (Valence, Arousal) coordinate point in Deezer dataset.
KD-Tree queries the unique 2D coordinate points, then retrieves candidate songs mapping to those points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors


class SongDatasetClass:
    """
    Manages the song database, coordinate mappings, and the KD-Tree for 2D V-A space.
    """
    def __init__(self, df: pd.DataFrame, audio_feature_cols: List[str], va_cols: Tuple[str, str] = ('valence', 'arousal')):
        """
        Args:
            df: Cleaned and preprocessed DataFrame containing track_id, song_title, artist_name, valence, arousal, and audio features.
            audio_feature_cols: List of 12 PCA feature column names (e.g. ['pca_0', ..., 'pca_11']).
            va_cols: Tuple of column names for Valence and Arousal.
        """
        self.df = df.copy().reset_index(drop=True)
        self.audio_feature_cols = audio_feature_cols
        self.val_col = va_cols[0]
        self.aro_col = va_cols[1]
        
        # Fast lookup dictionaries
        self.track_lookup: Dict[str, dict] = {}
        for _, row in self.df.iterrows():
            tid = str(row['track_id'])
            self.track_lookup[tid] = {
                'track_id': tid,
                'song_title': str(row.get('song_title', 'Unknown')),
                'artist_name': str(row.get('artist_name', 'Unknown')),
                'valence': float(row[self.val_col]),
                'arousal': float(row[self.aro_col]),
                'audio_vector': row[self.audio_feature_cols].values.astype(np.float64)
            }
            
        # Map each unique rounded coordinate (valence, arousal) to a list of track_ids
        # In Gaur's study, ~17,751 songs mapped to 2,762 unique coordinate points.
        self.coord_to_track_ids: Dict[Tuple[float, float], List[str]] = {}
        for _, row in self.df.iterrows():
            pt = (round(float(row[self.val_col]), 4), round(float(row[self.aro_col]), 4))
            if pt not in self.coord_to_track_ids:
                self.coord_to_track_ids[pt] = []
            self.coord_to_track_ids[pt].append(str(row['track_id']))
            
        # Unique coordinate points matrix for KD-Tree
        self.unique_coords = np.array(list(self.coord_to_track_ids.keys()), dtype=np.float64)
        
        # Fit KD-Tree on unique V-A coordinates
        self.kd_tree = NearestNeighbors(n_neighbors=min(50, len(self.unique_coords)), algorithm='kd_tree', metric='euclidean')
        self.kd_tree.fit(self.unique_coords)

    def get_song(self, track_id: str) -> Optional[dict]:
        """Retrieve song metadata and vectors by track_id."""
        return self.track_lookup.get(str(track_id))

    def get_random_song(self, seed: Optional[int] = None) -> dict:
        """Picks a random song from the dataset."""
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.randint(0, len(self.df))
        tid = str(self.df.iloc[idx]['track_id'])
        return self.track_lookup[tid]

    def get_song_in_quadrant(self, quadrant: int, seed: Optional[int] = None) -> dict:
        """
        Retrieves a song situated in a specific Russell V-A quadrant:
        Quadrant 1: Happy/Exuberant (Valence > 0, Arousal > 0)
        Quadrant 2: Angry/Anxious   (Valence < 0, Arousal > 0)
        Quadrant 3: Sad/Depressed   (Valence < 0, Arousal < 0)
        Quadrant 4: Calm/Relaxed    (Valence > 0, Arousal < 0)
        """
        if seed is not None:
            np.random.seed(seed)
            
        if quadrant == 1:
            subset = self.df[(self.df[self.val_col] >= 0.1) & (self.df[self.aro_col] >= 0.1)]
        elif quadrant == 2:
            subset = self.df[(self.df[self.val_col] <= -0.1) & (self.df[self.aro_col] >= 0.1)]
        elif quadrant == 3:
            subset = self.df[(self.df[self.val_col] <= -0.1) & (self.df[self.aro_col] <= -0.1)]
        elif quadrant == 4:
            subset = self.df[(self.df[self.val_col] >= 0.1) & (self.df[self.aro_col] <= -0.1)]
        else:
            raise ValueError(f"Invalid quadrant {quadrant}. Must be 1, 2, 3, or 4.")
            
        if len(subset) == 0:
            # Fallback to broader zero boundary
            if quadrant == 1:
                subset = self.df[(self.df[self.val_col] >= 0) & (self.df[self.aro_col] >= 0)]
            elif quadrant == 2:
                subset = self.df[(self.df[self.val_col] <= 0) & (self.df[self.aro_col] >= 0)]
            elif quadrant == 3:
                subset = self.df[(self.df[self.val_col] <= 0) & (self.df[self.aro_col] <= 0)]
            elif quadrant == 4:
                subset = self.df[(self.df[self.val_col] >= 0) & (self.df[self.aro_col] <= 0)]
                
        idx = np.random.choice(subset.index)
        tid = str(self.df.loc[idx, 'track_id'])
        return self.track_lookup[tid]


def generate_synthetic_dataset(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates a realistic synthetic dataset mimicking the Deezer + Spotify feature distributions
    so that unit tests and end-to-end baseline runs can execute immediately without external downloads.
    """
    rng = np.random.default_rng(random_state)
    
    # 2D Russell Emotion Space: Valence and Arousal in [-1, 1]
    # Cluster slightly into 4 quadrants to reflect real music distributions
    valence = rng.uniform(-0.95, 0.95, size=n_samples)
    arousal = rng.uniform(-0.95, 0.95, size=n_samples)
    
    # Raw audio features (mimicking 15 Spotify features: danceability, energy, tempo, loudness, etc.)
    audio_feat_names = [f"raw_audio_{i}" for i in range(15)]
    audio_matrix = rng.normal(loc=0.0, scale=1.5, size=(n_samples, 15))
    
    # Inject correlated outliers to test IQR discretization (> 5.5 IQR)
    outlier_indices = rng.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    audio_matrix[outlier_indices, :] *= 6.0
    
    data = {
        'track_id': [f"TRK_{i:06d}" for i in range(n_samples)],
        'song_title': [f"Song Title {i}" for i in range(n_samples)],
        'artist_name': [f"Artist {i % 50}" for i in range(n_samples)],
        'valence': valence,
        'arousal': arousal,
    }
    for idx, col in enumerate(audio_feat_names):
        data[col] = audio_matrix[:, idx]
        
    df = pd.DataFrame(data)
    return df, audio_feat_names
