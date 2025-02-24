# vision.py

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple

class FaceMeshProcessor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image: np.ndarray):
        return self.face_mesh.process(image)

    @staticmethod
    def extract_points(face_landmarks) -> np.ndarray:
        points_3d = []
        for lm in face_landmarks.landmark:
            x_3d = (lm.x - 0.5) * 3.0
            y_3d = -(lm.y - 0.5) * 2.0
            z_3d = -lm.z * 2.0 + 1.5
            points_3d.append([x_3d, y_3d, z_3d])
        return np.array(points_3d, dtype=np.float32)

    @staticmethod
    def densify_points(points: np.ndarray, connections: List[Tuple[int, int]], num_interpolations: int = 2) -> np.ndarray:
        densified = list(points)
        for i, j in connections:
            if i < len(points) and j < len(points):
                p1 = points[i]
                p2 = points[j]
                for k in range(1, num_interpolations + 1):
                    alpha = k / (num_interpolations + 1)
                    interp_point = (1 - alpha) * p1 + alpha * p2
                    densified.append(interp_point)
        return np.array(densified, dtype=np.float32)

class EmotionRecognition:
    @staticmethod
    def detect_emotions(face_landmarks) -> Tuple[float, float]:
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        face_width = x_max - x_min
        if face_width < 1e-6:
            return 0.0, 0.0

        def dist_2d(a, b):
            return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

        corner_left = face_landmarks.landmark[61]
        corner_right = face_landmarks.landmark[291]
        mouth_corner_dist = dist_2d(corner_left, corner_right)
        happiness = mouth_corner_dist / face_width

        top_lip = face_landmarks.landmark[13]
        bottom_lip = face_landmarks.landmark[14]
        mouth_open_dist = dist_2d(top_lip, bottom_lip)
        surprise = mouth_open_dist / face_width

        happiness = max(0.0, min(1.0, happiness))
        surprise = max(0.0, min(1.0, surprise))

        return happiness, surprise

class HandTracker:
    def __init__(self):
        self.hand_tracker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image: np.ndarray):
        return self.hand_tracker.process(image)

    @staticmethod
    def extract_points(hand_landmarks) -> np.ndarray:
        points_3d = []
        for lm in hand_landmarks.landmark:
            x_3d = (lm.x - 0.5) * 3.0
            y_3d = -(lm.y - 0.5) * 2.0
            z_3d = -lm.z * 2.0
            points_3d.append([x_3d, y_3d, z_3d])
        return np.array(points_3d, dtype=np.float32)
