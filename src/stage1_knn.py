"""
Stage 1: Linear Waypoint Interpolation and KD-Tree Nearest Neighbor Search
Paper / Thesis: Shaurya Gaur (2023), "Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"

Mathematical Formulation (Equation 5 in Gaur 2023):
    S_T = S_L + (S_N - S_L) / (N - L)

Where:
    S_T: Target waypoint vector in 2D Emotion Space (Valence, Arousal)
    S_L: Position of the last chosen song in the playlist (step L)
    S_N: Position of the destination song (final target at step N)
    N:   Total length of playlist
    L:   Current step index in playlist (L in [0, N-1])
"""

import numpy as np
from typing import List, Set, Tuple
from .dataset import SongDatasetClass


def calculate_target_waypoint(s_last_va: np.ndarray, s_dest_va: np.ndarray, current_step: int, total_steps: int) -> np.ndarray:
    """
    Computes S_T according to Equation 5 in Gaur (2023).
    
    Args:
        s_last_va: 2D array [valence, arousal] of the most recently added song.
        s_dest_va: 2D array [valence, arousal] of the target final destination song.
        current_step: Current playlist length (L) before adding the next song.
        total_steps: Desired total playlist length (N).
        
    Returns:
        target_point: 2D array [valence, arousal] representing S_T.
    """
    remaining_steps = total_steps - current_step
    if remaining_steps <= 0:
        return s_dest_va.copy()
        
    step_vector = (s_dest_va - s_last_va) / float(remaining_steps)
    target_point = s_last_va + step_vector
    return target_point


def query_stage1_knn(dataset: SongDatasetClass, 
                     target_va: np.ndarray, 
                     k_neighbors: int, 
                     visited_track_ids: Set[str], 
                     dest_track_id: str) -> List[str]:
    """
    Queries KD-Tree for nearest emotional coordinate points and gathers K candidate songs.
    
    Filters applied per Gaur (2023):
    1. No duplicate songs already present in playlist (visited_track_ids).
    2. Destination song S_N cannot be selected before step N.
    
    Args:
        dataset: SongDatasetClass instance holding KD-Tree and coordinate mappings.
        target_va: 2D target waypoint S_T [valence, arousal].
        k_neighbors: Desired number of candidate songs K (default 7).
        visited_track_ids: Set of track_ids already in the playlist.
        dest_track_id: track_id of the final destination song S_N.
        
    Returns:
        candidate_ids: List of candidate track_ids of length up to K.
    """
    candidate_ids: List[str] = []
    
    # Start querying nearest coordinates, expanding search radius if needed
    n_coord_query = min(len(dataset.unique_coords), max(k_neighbors * 3, 20))
    distances, indices = dataset.kd_tree.kneighbors([target_va], n_neighbors=n_coord_query)
    
    for coord_idx in indices[0]:
        coord_tuple = tuple(dataset.unique_coords[coord_idx])
        matching_songs = dataset.coord_to_track_ids.get(coord_tuple, [])
        
        for tid in matching_songs:
            # Filter duplicates and destination song
            if tid in visited_track_ids or tid == dest_track_id:
                continue
            candidate_ids.append(tid)
            if len(candidate_ids) >= k_neighbors:
                return candidate_ids
                
    # If still not enough candidates due to duplicates, expand query to entire unique coords
    if len(candidate_ids) < k_neighbors:
        distances, indices = dataset.kd_tree.kneighbors([target_va], n_neighbors=len(dataset.unique_coords))
        for coord_idx in indices[0]:
            coord_tuple = tuple(dataset.unique_coords[coord_idx])
            matching_songs = dataset.coord_to_track_ids.get(coord_tuple, [])
            for tid in matching_songs:
                if tid in visited_track_ids or tid == dest_track_id:
                    continue
                if tid not in candidate_ids:
                    candidate_ids.append(tid)
                if len(candidate_ids) >= k_neighbors:
                    return candidate_ids

    return candidate_ids
