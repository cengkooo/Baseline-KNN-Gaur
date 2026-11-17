"""
Two-Stage Mood-Dynamic Playlist Generator
Paper / Thesis: Shaurya Gaur (2023), "Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"

Orchestrates Stage 1 (linear waypoint interpolation + KD-Tree) and Stage 2 (audio distance re-ranking)
to generate playlists of length N connecting start song S_0 to destination song S_N.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .dataset import SongDatasetClass
from .stage1_knn import calculate_target_waypoint, query_stage1_knn
from .stage2_selector import select_best_candidate


class TwoStagePlaylistGenerator:
    """
    Main generator class reproducing Algorithm 1 in Gaur (2023).
    """
    def __init__(self, dataset: SongDatasetClass, k_neighbors: int = 7, metric: str = 'cosine'):
        """
        Args:
            dataset: Pre-indexed SongDatasetClass.
            k_neighbors: Number of emotion candidates to fetch at each step (default K=7).
            metric: Distance metric for Stage 2 audio re-ranking ('cosine', 'euclidean', 'manhattan').
        """
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.metric = metric

    def generate_playlist(self, start_track_id: str, dest_track_id: str, playlist_length: int = 12) -> pd.DataFrame:
        """
        Generates a playlist interpolating from start_track_id to dest_track_id.
        
        Args:
            start_track_id: Seed track ID (position 0).
            dest_track_id: Goal destination track ID (position N-1).
            playlist_length: Total desired length N (default 12).
            
        Returns:
            pd.DataFrame representing the sequential playlist with metadata and step metrics.
        """
        if start_track_id == dest_track_id:
            raise ValueError("Start song and destination song must be different.")
            
        start_song = self.dataset.get_song(start_track_id)
        dest_song = self.dataset.get_song(dest_track_id)
        
        if start_song is None:
            raise ValueError(f"Start track ID {start_track_id} not found in dataset.")
        if dest_song is None:
            raise ValueError(f"Destination track ID {dest_track_id} not found in dataset.")
            
        s_dest_va = np.array([dest_song['valence'], dest_song['arousal']], dtype=np.float64)
        
        # Initialize playlist with start song
        playlist_tracks: List[dict] = [start_song]
        visited_ids = {start_track_id}
        
        # Iteratively pick intermediate songs (L = 1 to N - 2)
        # Note: Indexing: Step 0 is start song, Step N-1 is destination song.
        # Intermediate steps count = playlist_length - 2
        for current_step in range(1, playlist_length - 1):
            last_song = playlist_tracks[-1]
            s_last_va = np.array([last_song['valence'], last_song['arousal']], dtype=np.float64)
            
            # 1. Compute target waypoint S_T (Equation 5)
            # Remaining steps = playlist_length - current_step
            target_va = calculate_target_waypoint(
                s_last_va=s_last_va,
                s_dest_va=s_dest_va,
                current_step=current_step,
                total_steps=playlist_length - 1
            )
            
            # 2. Stage 1: KD-Tree search on Valence-Arousal for K candidates
            candidates = query_stage1_knn(
                dataset=self.dataset,
                target_va=target_va,
                k_neighbors=self.k_neighbors,
                visited_track_ids=visited_ids,
                dest_track_id=dest_track_id
            )
            
            if not candidates:
                raise RuntimeError(f"No available candidates found at step {current_step}.")
                
            # 3. Stage 2: Audio distance re-ranking
            chosen_tid, dist = select_best_candidate(
                dataset=self.dataset,
                candidate_ids=candidates,
                reference_audio_vec=last_song['audio_vector'],
                metric=self.metric
            )
            
            chosen_song = self.dataset.get_song(chosen_tid)
            playlist_tracks.append(chosen_song)
            visited_ids.add(chosen_tid)
            
        # Append destination song at step N-1
        playlist_tracks.append(dest_song)
        
        # Build output DataFrame
        rows = []
        for step_idx, song in enumerate(playlist_tracks):
            row_dict = {
                'step': step_idx,
                'track_id': song['track_id'],
                'song_title': song['song_title'],
                'artist_name': song['artist_name'],
                'valence': song['valence'],
                'arousal': song['arousal']
            }
            # Add audio feature vectors
            for i, val in enumerate(song['audio_vector']):
                row_dict[self.dataset.audio_feature_cols[i]] = val
            rows.append(row_dict)
            
        return pd.DataFrame(rows)
