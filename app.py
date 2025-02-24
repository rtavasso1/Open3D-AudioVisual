import cv2
import mediapipe as mp
import numpy as np
import time
import math

from config import FINAL_WIDTH, FINAL_HEIGHT, OPENGGL_HEIGHT, WEBCAM_HEIGHT, DEFAULT_IP, DEFAULT_ABLETON_PORT, DEFAULT_TD_PORT
from image_utils import ImageUtils
from point_cloud import ConstellationGenerator, PointCloudEffects
from vision import FaceMeshProcessor, EmotionRecognition, HandTracker
from visualizer import Open3DVisualizer
from ableton_integration import AbletonController, HandLandmark

from pythonosc import udp_client

class FaceHandVisualizerApp:
    def __init__(self):
        self.face_processor = FaceMeshProcessor()
        self.hand_tracker = HandTracker()
        self.visualizer = Open3DVisualizer()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Generate the constellation star field with clusters.
        constellation_gen = ConstellationGenerator(num_points=1000, radius=2.0, z_scale=0.15,
                                                   min_size=0.5, max_size=2.5)
        (self.constellation_points, self.constellation_sizes,
         self.constellation_lines, self.constellation_cluster_count,
         self.cluster_info) = constellation_gen.generate()

        self.line_width = 2.0
        self.prev_time = time.time()
        self.color_phase = 0.0
        self.max_phase_speed = 2 * math.pi * 5.0

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        self.ableton = AbletonController(ip=DEFAULT_IP, port=DEFAULT_ABLETON_PORT)
        self.td_client = udp_client.SimpleUDPClient(DEFAULT_IP, DEFAULT_TD_PORT)

        # Track whether we are actively playing the session right now.
        self.is_playing = False

    def calculate_tempo(self, hand_landmarks) -> float:
        """Just an example of how you might map wrist Y-position to a BPM range."""
        wrist = hand_landmarks.landmark[HandLandmark.WRIST]
        inv_y = 1.0 - wrist.y
        min_tempo = 80
        max_tempo = 160
        return min_tempo + inv_y * (max_tempo - min_tempo)

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_processor.process(frame_rgb)
                hands_results = self.hand_tracker.process(frame_rgb)

                left_dispersion = 0.0
                palette_factor = 0.5

                # Flags to track whether we've detected these hands at all this frame
                left_present = False
                right_present = False

                hand_points_all = []
                if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                    for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks,
                                                          hands_results.multi_handedness):

                        # Draw the actual landmarks on the frame
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=hand_landmarks,
                            connections=mp.solutions.hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        # Extract positions for thumb and index
                        thumb = hand_landmarks.landmark[4]
                        index = hand_landmarks.landmark[8]
                        dx = thumb.x - index.x
                        dy = thumb.y - index.y
                        distance = math.sqrt(dx * dx + dy * dy)
                        # Normalize distance into 0..1
                        min_thresh = 0.05
                        max_thresh = 0.3
                        norm_distance = (distance - min_thresh) / (max_thresh - min_thresh)
                        norm_distance = max(0.0, min(norm_distance, 1.0))

                        label = handedness.classification[0].label
                        if label == "Left":
                            left_dispersion = norm_distance
                            left_present = True
                            self.ableton.update_rack_macro("space_rack", norm_distance)
                        elif label == "Right":
                            palette_factor = norm_distance
                            right_present = True
                            self.ableton.update_rack_macro("distort_rack", norm_distance)

                        # Draw a line between thumb and index finger tips
                        h, w, _ = frame.shape
                        thumb_point = (int(thumb.x * w), int(thumb.y * h))
                        index_point = (int(index.x * w), int(index.y * h))
                        cv2.line(frame, thumb_point, index_point, (255, 0, 0), 2)

                        # Display norm_distance near midpoint
                        mid_point = (int((thumb_point[0] + index_point[0]) / 2),
                                     int((thumb_point[1] + index_point[1]) / 2))
                        cv2.putText(frame, f"{norm_distance:.2f}", mid_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

                        # If you needed point-cloud data, you'd gather it here (omitted).
                        hand_points = np.array([])  # Not used in this example
                        hand_points_all.append(hand_points)
                else:
                    # If no hands, just pass empty data to the visualizer
                    self.visualizer.update_hands(np.array([]))

                # Update the 3D viewer with hand data (omitted or minimal for now)
                if hand_points_all:
                    all_hand_points = np.concatenate(hand_points_all, axis=0)
                    self.visualizer.update_hands(all_hand_points, color=[0, 1, 0])

                # Face mesh logic
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    points_3d = FaceMeshProcessor.extract_points(face_landmarks)
                    dense_points = FaceMeshProcessor.densify_points(
                        points_3d,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        num_interpolations=2
                    )
                    # Blend face points with the constellation
                    dispersed_points, dispersed_sizes = PointCloudEffects.constellation_dispersion(
                        dense_points,
                        self.constellation_points,
                        self.constellation_sizes,
                        left_dispersion,
                        face_size=2.0
                    )
                    # Downsample with star sizes
                    downsampled_points, downsampled_sizes = PointCloudEffects.downsample_with_sizes(
                        dispersed_points,
                        dispersed_sizes,
                        voxel_size=0.005
                    )
                    depth_colors = PointCloudEffects.compute_depth_colors(downsampled_points,
                                                                          self.color_phase,
                                                                          palette_factor,
                                                                          depth_freq=2.0)
                    final_points = PointCloudEffects.jitter(downsampled_points,
                                                            palette_factor * 0.05)
                    self.visualizer.update_face(final_points, depth_colors, downsampled_sizes)
                    self.visualizer.update_dummy_mesh(final_points)

                    # Update constellation lines for the cluster stars
                    if dense_points.shape[0] > 0 and self.constellation_cluster_count > 0:
                        if dense_points.shape[0] < self.constellation_cluster_count:
                            times = (self.constellation_cluster_count // dense_points.shape[0]) + 1
                            face_subset = np.tile(dense_points, (times, 1))[:self.constellation_cluster_count]
                        else:
                            face_subset = dense_points[:self.constellation_cluster_count]
                        dispersed_cluster_points, _ = PointCloudEffects.constellation_dispersion(
                            face_subset,
                            self.constellation_points[:self.constellation_cluster_count],
                            self.constellation_sizes[:self.constellation_cluster_count],
                            left_dispersion,
                            face_size=2.0
                        )
                        jittered_cluster_points = PointCloudEffects.jitter_full(
                            dispersed_cluster_points,
                            palette_factor * 0.05
                        )
                        self.visualizer.update_constellation_lines(jittered_cluster_points,
                                                                   self.constellation_lines,
                                                                   self.line_width,
                                                                   left_dispersion)
                        # Compute & update per-cluster labels
                        labels = []
                        for info in self.cluster_info:
                            s_idx = info["start_idx"]
                            e_idx = info["end_idx"]
                            if s_idx < jittered_cluster_points.shape[0] and e_idx < jittered_cluster_points.shape[0]:
                                cluster_pts = jittered_cluster_points[s_idx:e_idx+1]
                                center = np.mean(cluster_pts, axis=0)
                                labels.append({"label": info["label"],
                                               "position": center,
                                               "opacity": left_dispersion})
                        self.visualizer.update_constellation_labels(labels)
                    else:
                        self.visualizer.update_constellation_lines(np.array([]), [], self.line_width, 0.0)
                        self.visualizer.update_constellation_labels([])
                else:
                    self.visualizer.update_face(np.array([]), np.array([]), np.array([]))
                    self.visualizer.update_constellation_lines(np.array([]), [], self.line_width, 0.0)
                    self.visualizer.update_constellation_labels([])

                # Handle color cycling
                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time
                default_speed_factor = 0.2
                self.color_phase += dt * self.max_phase_speed * default_speed_factor

                # 3D render output
                opengl_img = self.visualizer.render()
                opengl_img = cv2.cvtColor(opengl_img, cv2.COLOR_RGB2BGR)
                opengl_img_resized = cv2.resize(opengl_img, (FINAL_WIDTH, OPENGGL_HEIGHT))

                webcam_img_processed = ImageUtils.resize_and_crop(frame, FINAL_WIDTH, WEBCAM_HEIGHT)
                unified_view = np.vstack([opengl_img_resized, webcam_img_processed])
                cv2.imshow('Unified View', unified_view)

                #
                # --- NEW: Hand-driven playback logic ---
                #
                hands_present = left_present or right_present

                if hands_present:
                    # If not currently playing, start the timeline and fire clips once
                    if not self.is_playing:
                        # Fire all available clips (one per track) so they loop
                        for t_name, track in self.ableton.tracks.items():
                            first_clip_idx = track.clips[0].index
                            self.ableton.play_clip(track.index, first_clip_idx)
                        # Then start the global playback
                        self.ableton.start_song_playback()
                        self.is_playing = True

                    # Mute/unmute tracks according to which hand is present
                    # Left hand -> 'other' & 'vocal'
                    self.ableton.set_track_mute("other", 0 if left_present else 1)
                    self.ableton.set_track_mute("vocal", 0 if left_present else 1)
                    # Right hand -> 'bass' & 'drums'
                    self.ableton.set_track_mute("Bass", 0 if right_present else 1)
                    self.ableton.set_track_mute("drums", 0 if right_present else 1)

                else:
                    # No hands present -> Stop playback if we are currently playing
                    if self.is_playing:
                        self.ableton.stop_song_playback()
                        self.is_playing = False

                # Exit on ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.visualizer.close()

def main():
    app = FaceHandVisualizerApp()
    app.run()

if __name__ == '__main__':
    main()
