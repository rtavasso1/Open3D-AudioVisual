# point_cloud.py

import numpy as np
import math
import open3d as o3d
from scipy.spatial import cKDTree

class ConstellationGenerator:
    def __init__(self, num_points: int = 5000, radius: float = 2.5, z_scale: float = 0.15,
                 min_size: float = 0.5, max_size: float = 2.5,
                 num_clusters: int = 5, cluster_min: int = 5, cluster_max: int = 15):
        """
        Initializes the generator for a starry sky with constellation effects.
        """
        self.num_points = num_points
        self.radius = radius
        self.z_scale = z_scale
        self.min_size = min_size
        self.max_size = max_size
        self.num_clusters = num_clusters
        self.cluster_min = cluster_min
        self.cluster_max = cluster_max

    def generate(self) -> (np.ndarray, np.ndarray, list, int, list):
        """
        Generates a starry sky with two components:
        1. Constellation clusters – each cluster’s stars are placed along a baseline 
           with an added sinusoidal perturbation. Consecutive stars in each cluster 
           are connected by lines.
        2. Background stars – uniformly distributed over a disk.
        
        Returns:
            points: (N,3) array of star positions.
            sizes: (N,) array of star sizes.
            lines: List of (i, j) pairs connecting cluster star indices.
            cluster_count: Total number of cluster stars.
            cluster_info: A list of dicts, one per cluster, each with keys:
                          "start_idx" (first index in points),
                          "end_idx" (last index in points),
                          "label" (the constellation name for that cluster).
        """
        cluster_points = []
        cluster_sizes = []
        cluster_lines = []
        cluster_info = []
        total_cluster_stars = 0

        # Define a list of famous constellation names (cycle if needed).
        famous_names = ['Orion', 'Cassiopeia', 'Ursa Major', 'Cygnus', 'Scorpius',
                        'Leo', 'Taurus', 'Gemini', 'Aquila']
        name_idx = 0

        for _ in range(self.num_clusters):
            # Random center within disk.
            r = self.radius * math.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * math.pi)
            center = np.array([r * math.cos(theta), r * math.sin(theta), np.random.normal(scale=self.z_scale)])
            
            # Number of stars in this cluster.
            n_stars = np.random.randint(self.cluster_min, self.cluster_max + 1)
            start_idx = len(cluster_points)
            
            # Choose a baseline angle.
            line_angle = np.random.uniform(0, 2 * math.pi)
            spacing = 0.1  # base spacing

            # Sinusoidal perturbation parameters.
            amplitude = np.random.uniform(0.05, 0.15)
            frequency = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * math.pi)
            perp = np.array([-math.sin(line_angle), math.cos(line_angle), 0])
            
            for j in range(n_stars):
                offset = j * spacing
                base_offset = np.array([offset * math.cos(line_angle),
                                         offset * math.sin(line_angle),
                                         0])
                sine_offset = amplitude * math.sin(j * frequency + phase) * perp
                star_offset = base_offset + sine_offset
                star = center + star_offset
                cluster_points.append(star.tolist())
                star_size = np.random.uniform(self.min_size, self.max_size)
                cluster_sizes.append(star_size)
                if j > 0:
                    cluster_lines.append((start_idx + j - 1, start_idx + j))
            end_idx = len(cluster_points) - 1
            # Assign a label from the famous names list.
            label = famous_names[name_idx % len(famous_names)]
            name_idx += 1
            cluster_info.append({"start_idx": start_idx, "end_idx": end_idx, "label": label})
            total_cluster_stars += n_stars

        # Generate background stars.
        background_points = []
        background_sizes = []
        background_count = max(self.num_points - total_cluster_stars, 0)
        for _ in range(background_count):
            r = self.radius * math.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * math.pi)
            background_points.append([r * math.cos(theta), r * math.sin(theta), np.random.normal(scale=self.z_scale)])
            star_size = np.random.uniform(self.min_size, self.max_size)
            background_sizes.append(star_size)

        points = np.array(cluster_points + background_points, dtype=np.float32)
        sizes = np.array(cluster_sizes + background_sizes, dtype=np.float32)
        return points, sizes, cluster_lines, total_cluster_stars, cluster_info


class PointCloudEffects:
    @staticmethod
    def constellation_dispersion(face_points: np.ndarray, constellation_points: np.ndarray, 
                                 constellation_sizes: np.ndarray, factor: float, face_size: float = 2.0) -> (np.ndarray, np.ndarray):
        """
        Blends face points with constellation star positions.
        """
        if face_points.shape[0] == 0:
            return face_points, np.array([])
        n_face = face_points.shape[0]
        n_const = constellation_points.shape[0]
        if n_face > n_const:
            times = (n_face // n_const) + 1
            big_constellation = np.tile(constellation_points, (times, 1))
            big_sizes = np.tile(constellation_sizes, times)
            subset_points = big_constellation[:n_face]
            subset_sizes = big_sizes[:n_face]
        else:
            subset_points = constellation_points[:n_face]
            subset_sizes = constellation_sizes[:n_face]
        dispersed_points = (1 - factor) * face_points + factor * subset_points
        dispersed_sizes = (1 - factor) * face_size + factor * subset_sizes
        return dispersed_points, dispersed_sizes

    @staticmethod
    def jitter(points: np.ndarray, jitter_intensity: float) -> np.ndarray:
        """
        Applies jitter only along the x-axis.
        """
        jittered_points = points.copy()
        noise = np.random.normal(scale=jitter_intensity, size=(points.shape[0], 1))
        jittered_points[:, 0] += noise[:, 0]
        return jittered_points

    @staticmethod
    def jitter_full(points: np.ndarray, jitter_intensity: float) -> np.ndarray:
        """
        Applies jitter along all axes.
        """
        jittered_points = points.copy()
        noise = np.random.normal(scale=jitter_intensity, size=points.shape)
        jittered_points += noise
        return jittered_points

    @staticmethod
    def inversion(points: np.ndarray, t: float, axis: str = 'z') -> np.ndarray:
        """
        Inverts points along the specified axis based on a sine modulation.
        """
        axis = axis.lower()
        if axis == 'x':
            axis_idx = 0
        elif axis == 'y':
            axis_idx = 1
        elif axis == 'z':
            axis_idx = 2
        else:
            raise ValueError("Invalid axis string. Choose 'x', 'y', or 'z'.")
        inversion_amount = (math.sin(t) + 1) / 2
        new_points = points.copy()
        axis_vals = points[:, axis_idx]
        min_val = np.min(axis_vals)
        max_val = np.max(axis_vals)
        inverted_vals = min_val + max_val - axis_vals
        new_points[:, axis_idx] = (1 - inversion_amount) * axis_vals + inversion_amount * inverted_vals
        return new_points

    @staticmethod
    def downsample(points: np.ndarray, voxel_size: float = 0.005) -> np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        down_pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(down_pcd.points, dtype=np.float32)

    @staticmethod
    def downsample_with_sizes(points: np.ndarray, sizes: np.ndarray, voxel_size: float = 0.005) -> (np.ndarray, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        down_pcd = pcd.voxel_down_sample(voxel_size)
        down_points = np.asarray(down_pcd.points, dtype=np.float32)
        if down_points.shape[0] == 0:
            return down_points, sizes
        tree = cKDTree(points)
        _, indices = tree.query(down_points)
        down_sizes = sizes[indices]
        return down_points, down_sizes

    @staticmethod
    def compute_depth_colors(points: np.ndarray, phase_offset: float, palette_factor: float, depth_freq: float = 2.0) -> np.ndarray:
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        z_vals = points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = max(z_max - z_min, 1e-6)
        colors = []
        for z in z_vals:
            d = (z - z_min) / z_range
            phase = phase_offset - (2 * math.pi * depth_freq * d)
            wave_val = 0.5 + 0.5 * math.sin(phase)
            # Palette A: near-white to fluorescent magenta.
            r_a = wave_val
            g_a = (1 - d) * 0.8 * wave_val
            b_a = wave_val
            # Palette B: cyan to yellow.
            r_b = d * wave_val
            g_b = wave_val
            b_b = (1 - d) * wave_val
            r = (1 - palette_factor) * r_a + palette_factor * r_b
            g = (1 - palette_factor) * g_a + palette_factor * g_b
            b = (1 - palette_factor) * b_a + palette_factor * b_b
            colors.append([r, g, b])
        return np.array(colors, dtype=np.float32)
