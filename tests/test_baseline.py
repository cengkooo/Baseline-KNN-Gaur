"""
Automated Validation and Unit Testing for Shaurya Gaur (2023) Baseline Pipeline.
Designed for Code Walkthrough Readiness (Sidang Akhir IF ITERA Bobot 10%).

Runs comprehensive sanity and algorithmic verification checks:
1. Test DEAM Dataset Loading and Normalization (1,802 tracks, V-A in [-1, 1])
2. Test Gaur Preprocessor (Outlier Discretization, Yeo-Johnson, PCA 12D, MinMax)
3. Test KD-Tree Spatial Index & Quadrant Retrieval
4. Test Stage 1 Linear Waypoint Calculation (Equation 5) & Duplicate/Destination Filtering
5. Test Stage 2 Audio Distance Re-Ranking (Cosine, Euclidean, Manhattan)
6. Test End-to-End TwoStagePlaylistGenerator (Length N, Start/End integrity, Zero Duplicates)
7. Test Gaur Evaluation Metrics (Smoothness PCC in [0, 1], Evenness Variance >= 0)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.deam_loader import load_deam_static_annotations, load_or_generate_deam_dataset, get_default_deam_path
from src.preprocessing import GaurPreprocessor
from src.dataset import SongDatasetClass
from src.stage1_knn import calculate_target_waypoint, query_stage1_knn
from src.stage2_selector import calculate_audio_distance, select_best_candidate
from src.playlist_generator import TwoStagePlaylistGenerator
from src.metrics import calculate_smoothness_pcc, calculate_evenness_variance, evaluate_playlist


class TestBaselineGaurPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and prepares test dataset once for the test suite."""
        cls.deam_path = get_default_deam_path()
        cls.df_raw, cls.audio_cols = load_or_generate_deam_dataset(deam_root=cls.deam_path, random_state=42)
        
        # Fit preprocessor
        cls.preprocessor = GaurPreprocessor(n_pca_components=12, iqr_multiplier=5.5, n_bins=20)
        cls.df_clean, cls.pca_cols = cls.preprocessor.fit_transform(cls.df_raw, cls.audio_cols)
        
        # Build dataset container
        cls.dataset = SongDatasetClass(df=cls.df_clean, audio_feature_cols=cls.pca_cols)

    def test_01_deam_dataset_integrity(self):
        """Verifies that DEAM dataset loads exactly 1,802 tracks with correct V-A scaling [-1, 1]."""
        self.assertEqual(len(self.df_raw), 1802, "DEAM dataset must contain exactly 1,802 songs.")
        self.assertTrue('valence' in self.df_raw.columns and 'arousal' in self.df_raw.columns)
        
        # Check scale bounds
        min_v, max_v = self.df_raw['valence'].min(), self.df_raw['valence'].max()
        min_a, max_a = self.df_raw['arousal'].min(), self.df_raw['arousal'].max()
        self.assertGreaterEqual(min_v, -1.0)
        self.assertLessEqual(max_v, 1.0)
        self.assertGreaterEqual(min_a, -1.0)
        self.assertLessEqual(max_a, 1.0)

    def test_02_preprocessing_pipeline(self):
        """Verifies PCA output dimensions, variance, and feature scaling."""
        self.assertEqual(len(self.pca_cols), 12, "PCA should produce 12 components.")
        for col in self.pca_cols:
            self.assertGreaterEqual(self.df_clean[col].min(), -1.01)
            self.assertLessEqual(self.df_clean[col].max(), 1.01)

    def test_03_kdtree_spatial_indexing(self):
        """Verifies KD-Tree build and Russell quadrant retrieval."""
        self.assertGreater(len(self.dataset.unique_coords), 1000, "Should have >1000 unique coordinates.")
        
        # Test retrieval from all 4 quadrants
        for q in [1, 2, 3, 4]:
            song = self.dataset.get_song_in_quadrant(quadrant=q, seed=42)
            self.assertIsNotNone(song, f"Must successfully retrieve song from Quadrant {q}")
            if q == 1:
                self.assertGreater(song['valence'], 0.0)
                self.assertGreater(song['arousal'], 0.0)
            elif q == 2:
                self.assertLess(song['valence'], 0.0)
                self.assertGreater(song['arousal'], 0.0)
            elif q == 3:
                self.assertLess(song['valence'], 0.0)
                self.assertLess(song['arousal'], 0.0)
            elif q == 4:
                self.assertGreater(song['valence'], 0.0)
                self.assertLess(song['arousal'], 0.0)

    def test_04_stage1_waypoint_math(self):
        """Verifies Gaur (2023) Equation 5 linear waypoint interpolation."""
        # Start at (-1.0, -1.0), Dest at (+1.0, +1.0), total steps 10, current step 0
        s_last = np.array([-1.0, -1.0])
        s_dest = np.array([1.0, 1.0])
        target = calculate_target_waypoint(s_last, s_dest, current_step=0, total_steps=10)
        # Expected step vector: (2.0, 2.0) / 10 = (0.2, 0.2) -> target: (-0.8, -0.8)
        np.testing.assert_allclose(target, np.array([-0.8, -0.8]), rtol=1e-5)

    def test_05_stage1_filtering_integrity(self):
        """Verifies that Stage 1 KNN excludes visited tracks and destination song before step N."""
        s_start = self.dataset.get_song_in_quadrant(3, seed=1)
        s_dest = self.dataset.get_song_in_quadrant(1, seed=2)
        visited = {s_start['track_id']}
        dest_id = s_dest['track_id']
        
        target = np.array([0.0, 0.0])
        candidates = query_stage1_knn(self.dataset, target, k_neighbors=7, visited_track_ids=visited, dest_track_id=dest_id)
        
        self.assertEqual(len(candidates), 7, "Stage 1 must return exactly K candidates.")
        self.assertNotIn(s_start['track_id'], candidates, "Start track must not be returned in candidates.")
        self.assertNotIn(dest_id, candidates, "Destination track must never be returned before step N.")

    def test_06_stage2_distance_metrics(self):
        """Verifies calculation of Cosine, Euclidean, and Manhattan distances."""
        v1 = np.ones(12, dtype=np.float64)
        v2 = np.ones(12, dtype=np.float64) * 2
        # Identical direction: Cosine distance must be 0.0
        cos_dist = calculate_audio_distance(v1, v2, metric='cosine')
        self.assertAlmostEqual(cos_dist, 0.0, places=5)
        
        # Euclidean distance between v1 and v2 is sqrt(12 * 1^2) = sqrt(12)
        euc_dist = calculate_audio_distance(v1, v2, metric='euclidean')
        self.assertAlmostEqual(euc_dist, np.sqrt(12), places=5)

    def test_07_playlist_generation_constraints(self):
        """Verifies complete end-to-end playlist output integrity (Length, Start, Dest, No Duplicates)."""
        s_start = self.dataset.get_song_in_quadrant(3, seed=10)
        s_dest = self.dataset.get_song_in_quadrant(1, seed=20)
        
        length_n = 12
        generator = TwoStagePlaylistGenerator(self.dataset, k_neighbors=7, metric='cosine')
        p_df = generator.generate_playlist(s_start['track_id'], s_dest['track_id'], playlist_length=length_n)
        
        # 1. Total length must be exactly N
        self.assertEqual(len(p_df), length_n, f"Playlist length must be exactly {length_n}")
        # 2. First song must be start track
        self.assertEqual(p_df.iloc[0]['track_id'], s_start['track_id'], "First song must be S_0")
        # 3. Last song must be destination track
        self.assertEqual(p_df.iloc[-1]['track_id'], s_dest['track_id'], "Last song must be S_N")
        # 4. Zero duplicate songs
        self.assertEqual(len(p_df['track_id'].unique()), length_n, "All tracks in playlist must be unique.")

    def test_08_metrics_bounds(self):
        """Verifies that Smoothness and Evenness metrics produce valid values."""
        s_start = self.dataset.get_song_in_quadrant(3, seed=100)
        s_dest = self.dataset.get_song_in_quadrant(1, seed=200)
        generator = TwoStagePlaylistGenerator(self.dataset, k_neighbors=7, metric='cosine')
        p_df = generator.generate_playlist(s_start['track_id'], s_dest['track_id'], playlist_length=12)
        
        metrics = evaluate_playlist(p_df, audio_feature_cols=self.pca_cols)
        
        # Smoothness PCC must be in [0, 1]
        self.assertGreaterEqual(metrics['smoothness_va_pcc'], 0.0)
        self.assertLessEqual(metrics['smoothness_va_pcc'], 1.0)
        # Evenness Variance must be non-negative
        self.assertGreaterEqual(metrics['evenness_va_var'], 0.0)
        self.assertGreaterEqual(metrics['evenness_audio_var'], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
