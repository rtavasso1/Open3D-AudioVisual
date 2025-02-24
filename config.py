# config.py

FINAL_WIDTH = 720
FINAL_HEIGHT = 1280
OPENGGL_HEIGHT = int(FINAL_HEIGHT * 0.7)  # Top 70% for OpenGL render.
WEBCAM_HEIGHT = FINAL_HEIGHT - OPENGGL_HEIGHT  # Bottom 30% for webcam feed.

DEFAULT_IP = "127.0.0.1"
DEFAULT_ABLETON_PORT = 11000
DEFAULT_TD_PORT = 11001
