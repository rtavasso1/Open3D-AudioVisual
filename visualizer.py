# visualizer.py

import open3d as o3d
import numpy as np

from config import FINAL_WIDTH, OPENGGL_HEIGHT

class Open3DVisualizer:
    def __init__(self, window_width: int = FINAL_WIDTH, window_height: int = OPENGGL_HEIGHT):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Face and Hand Point Cloud',
                               width=window_width,
                               height=window_height,
                               visible=False)
        render_option = self.vis.get_render_option()
        render_option.point_size = 5.0
        render_option.background_color = np.array([0, 0, 0])
        
        self.pcd_face = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_face)
        
        self.pcd_hands = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_hands)
        
        self.dummy_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.1)
        self.dummy_mesh.paint_uniform_color([0, 0, 0])
        self.vis.add_geometry(self.dummy_mesh)
        
        self.line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.line_set)
        
        # Field for per-cluster constellation labels:
        self.constellation_labels = []  # List of dicts with keys: "label", "position", "opacity"

    def update_constellation_labels(self, labels: list):
        """
        Updates the per-cluster constellation labels.
        
        Parameters:
            labels: List of dicts, each with keys:
                    "label"    - constellation name (string)
                    "position" - 3D position (numpy array) in world coordinates
                    "opacity"  - Opacity factor (0 to 1)
        """
        self.constellation_labels = labels

    def update_face(self, points: np.ndarray, colors: np.ndarray, sizes: np.ndarray):
        """
        Updates the face point cloud geometry.
        """
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float32)
        if colors.size == 0:
            colors = np.empty((0, 3), dtype=np.float32)
        self.pcd_face.points = o3d.utility.Vector3dVector(points)
        self.pcd_face.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd_face)

    def update_hands(self, points: np.ndarray, color: list = [0, 1, 0]):
        """
        Updates the hand point cloud geometry.
        """
        if points.size > 0:
            self.pcd_hands.points = o3d.utility.Vector3dVector(points)
            colors = np.tile(np.array(color), (points.shape[0], 1))
            self.pcd_hands.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd_hands)
        else:
            self.pcd_hands.points = o3d.utility.Vector3dVector([])
            self.vis.update_geometry(self.pcd_hands)

    def update_dummy_mesh(self, points: np.ndarray):
        """
        Updates the dummy mesh geometry based on the bounds of the provided points.
        """
        if points.shape[0] == 0:
            return
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        bbox_size = max_bounds - min_bounds
        scale_factor = np.linalg.norm(bbox_size) * 0.5
        # Scale and translate the dummy mesh to be centered within the bounds.
        self.dummy_mesh.scale(scale_factor, center=self.dummy_mesh.get_center())
        self.dummy_mesh.translate(min_bounds + bbox_size / 2 - self.dummy_mesh.get_center())
        self.vis.update_geometry(self.dummy_mesh)

    def update_constellation_lines(self, points: np.ndarray, line_indices: list, line_width: float, opacity: float):
        """
        Updates the constellation line connections based on the provided points and connectivity info.

        Parameters:
            points: (N, 3) numpy array of blended cluster star positions.
            line_indices: List of (i, j) tuples indicating which points to connect.
            line_width: Tunable width for the drawn lines.
            opacity: Opacity factor (0 to 1) controlling line brightness; 0 makes lines invisible.
        """
        if points.size == 0 or len(line_indices) == 0 or opacity <= 0:
            self.line_set.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            self.line_set.lines = o3d.utility.Vector2iVector([])
            self.line_set.colors = o3d.utility.Vector3dVector([])
            self.vis.update_geometry(self.line_set)
            return

        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(line_indices)
        color = [opacity, opacity, opacity]
        colors = [color for _ in range(len(line_indices))]
        self.line_set.colors = o3d.utility.Vector3dVector(colors)
        render_option = self.vis.get_render_option()
        render_option.line_width = line_width
        self.vis.update_geometry(self.line_set)

    def render(self) -> np.ndarray:
        """
        Renders the scene and overlays each constellation label at its corresponding screen position.
        """
        self.vis.poll_events()
        self.vis.update_renderer()
        img = np.asarray(self.vis.capture_screen_float_buffer(False))
        img = (img * 255).astype(np.uint8)

        # Retrieve camera parameters from the current Open3D view.
        vc = self.vis.get_view_control()
        cam_params = vc.convert_to_pinhole_camera_parameters()
        intrinsic = cam_params.intrinsic.intrinsic_matrix  # 3x3
        extrinsic = cam_params.extrinsic                  # 4x4

        def project_point(pt_3d):
            """
            Projects a 3D point in world space to 2D screen coordinates based on
            the current Open3D view's intrinsic and extrinsic camera parameters.
            Returns None if the point is behind the camera.
            """
            # Convert the point to homogeneous coordinates.
            pt_3d_h = np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0])
            # Transform into camera coordinates.
            camera_coords = extrinsic @ pt_3d_h
            
            # If z <= 0, it's behind the camera (can't project).
            if camera_coords[2] <= 0:
                return None
            
            # Pinhole camera projection:
            #   X_pixel = fx*(x/z) + cx
            #   Y_pixel = fy*(y/z) + cy
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            x_2d = (camera_coords[0] / camera_coords[2]) * fx + cx
            y_2d = (camera_coords[1] / camera_coords[2]) * fy + cy

            return (int(round(x_2d)), int(round(y_2d)))

        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX

        for label_info in self.constellation_labels:
            text = label_info["label"]
            opacity = label_info["opacity"]
            if opacity < 0.05:
                # Skip near-zero opacity.
                continue
            
            pt_screen = project_point(label_info["position"])
            if pt_screen is None:
                # Point behind the camera or invalid.
                continue

            # Scale font and color according to opacity.
            max_font_scale = 0.75
            scaled_font = opacity * max_font_scale
            scaled_thickness = max(1, int(round(opacity * 1)))
            color = (int(255 * opacity), int(255 * opacity), int(255 * opacity))

            # Small offset if desired (slight nudge to the right).
            offset_x = int(20 * opacity)
            x = pt_screen[0] + offset_x
            y = pt_screen[1]

            cv2.putText(img, text, (x, y), font, scaled_font, color, scaled_thickness, cv2.LINE_AA)

        return img


    def close(self):
        self.vis.destroy_window()
