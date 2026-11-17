"""
Baseline Gaur (2023) Package Initialization.
"""

from .preprocessing import GaurPreprocessor
from .dataset import SongDatasetClass, generate_synthetic_dataset
from .stage1_knn import calculate_target_waypoint, query_stage1_knn
from .stage2_selector import calculate_audio_distance, select_best_candidate
from .playlist_generator import TwoStagePlaylistGenerator
from .metrics import calculate_smoothness_pcc, calculate_evenness_variance, evaluate_playlist
from .deam_loader import load_deam_static_annotations, load_or_generate_deam_dataset, load_deam_dynamic_annotations
from .utils_cuda import get_torch_device, log_vram_usage

__all__ = [
    'GaurPreprocessor',
    'SongDatasetClass',
    'generate_synthetic_dataset',
    'calculate_target_waypoint',
    'query_stage1_knn',
    'calculate_audio_distance',
    'select_best_candidate',
    'TwoStagePlaylistGenerator',
    'calculate_smoothness_pcc',
    'calculate_evenness_variance',
    'evaluate_playlist',
    'load_deam_static_annotations',
    'load_or_generate_deam_dataset',
    'load_deam_dynamic_annotations',
    'get_torch_device',
    'log_vram_usage'
]
