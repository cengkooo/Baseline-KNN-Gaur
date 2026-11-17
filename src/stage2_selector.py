"""
Stage 2: Audio Distance Re-Ranking and Selection
Paper / Thesis: Shaurya Gaur (2023), "Mood-Dynamic Playlist: Interpolating a Path of Emotions Using a KNN Algorithm"

Given K candidate songs from Stage 1 (KD-Tree on Valence-Arousal), Stage 2 ranks them
based on acoustic similarity in the 12-dimensional PCA audio feature space.

Supported distance metrics tested by Gaur:
- 'cosine' (Cosine Distance = 1 - Cosine Similarity) -> Baseline Winner
- 'euclidean' (L2 Euclidean Distance)
- 'manhattan' (L1 Cityblock Distance)

Hardware Acceleration:
Supports PyTorch CUDA batch tensor operations on NVIDIA GeForce RTX 3070 Ti,
with graceful fallback to NumPy / Scipy when running on CPU.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean, cityblock

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .dataset import SongDatasetClass


def calculate_audio_distance(vec_a: np.ndarray, vec_b: np.ndarray, metric: str = 'cosine') -> float:
    """
    Calculates distance between two 12-dimensional PCA audio feature vectors using Scipy.
    """
    metric_lower = metric.lower()
    if metric_lower == 'cosine':
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return float(cosine(vec_a, vec_b))
    elif metric_lower == 'euclidean':
        return float(euclidean(vec_a, vec_b))
    elif metric_lower == 'manhattan':
        return float(cityblock(vec_a, vec_b))
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'cosine', 'euclidean', or 'manhattan'.")


def select_best_candidate_torch(dataset: SongDatasetClass, 
                                candidate_ids: List[str], 
                                reference_audio_vec: np.ndarray, 
                                metric: str = 'cosine',
                                device: Optional[str] = None) -> Tuple[str, float]:
    """
    PyTorch CUDA accelerated batch candidate distance ranking on NVIDIA RTX 3070 Ti.
    """
    if device is None:
        device = 'cuda:0' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        
    cand_vectors = []
    valid_ids = []
    for tid in candidate_ids:
        song = dataset.get_song(tid)
        if song is not None:
            cand_vectors.append(song['audio_vector'])
            valid_ids.append(tid)
            
    if not valid_ids:
        raise ValueError("No valid candidates found.")
        
    cands_tensor = torch.tensor(np.array(cand_vectors), dtype=torch.float32, device=device)
    ref_tensor = torch.tensor(reference_audio_vec, dtype=torch.float32, device=device).unsqueeze(0)
    
    metric_lower = metric.lower()
    if metric_lower == 'cosine':
        # Cosine distance: 1.0 - cosine_similarity
        sims = F.cosine_similarity(ref_tensor, cands_tensor, dim=-1)
        dists = 1.0 - sims
    elif metric_lower == 'euclidean':
        dists = torch.cdist(ref_tensor, cands_tensor, p=2.0).squeeze(0)
    elif metric_lower == 'manhattan':
        dists = torch.cdist(ref_tensor, cands_tensor, p=1.0).squeeze(0)
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")
        
    best_idx = int(torch.argmin(dists).item())
    best_dist = float(dists[best_idx].item())
    return valid_ids[best_idx], best_dist


def select_best_candidate(dataset: SongDatasetClass, 
                           candidate_ids: List[str], 
                           reference_audio_vec: np.ndarray, 
                           metric: str = 'cosine',
                           use_torch: bool = True) -> Tuple[str, float]:
    """
    Ranks candidates by distance to the reference audio vector, selecting the minimum distance.
    Automatically leverages PyTorch on CUDA if available.
    """
    if not candidate_ids:
        raise ValueError("Candidate list is empty. Cannot select best candidate.")
        
    if use_torch and TORCH_AVAILABLE:
        try:
            return select_best_candidate_torch(dataset, candidate_ids, reference_audio_vec, metric=metric)
        except Exception:
            # Fallback to numpy on any tensor error
            pass
            
    # NumPy / Scipy fallback
    best_tid = candidate_ids[0]
    best_dist = float('inf')
    
    for tid in candidate_ids:
        song = dataset.get_song(tid)
        if song is None:
            continue
        cand_vec = song['audio_vector']
        dist = calculate_audio_distance(reference_audio_vec, cand_vec, metric=metric)
        
        if dist < best_dist:
            best_dist = dist
            best_tid = tid
            
    return best_tid, best_dist
