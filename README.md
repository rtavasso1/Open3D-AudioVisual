# Open3D-AudioVisual

## Overview

Open3D-AudioVisual is a project that integrates Open3D for visualization and MediaPipe for face and hand tracking. It also integrates with Ableton Live to control music playback and effects using hand gestures.

## Purpose

The purpose of this project is to create an interactive audiovisual experience by combining 3D visualization, face and hand tracking, and music control.

## Main Features

- 3D visualization using Open3D
- Face and hand tracking using MediaPipe
- Integration with Ableton Live for music control
- Real-time interaction and control using hand gestures

## Setup and Run Instructions

### Dependencies

- Python 3.8 or higher
- Open3D
- MediaPipe
- OpenCV
- numpy
- scipy
- python-osc

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rtavasso1/Open3D-AudioVisual.git
   cd Open3D-AudioVisual
   ```

2. Install the required dependencies:
   ```bash
   pip install open3d mediapipe opencv-python numpy scipy python-osc
   ```

### Configuration

Configuration settings are defined in the `config.py` file. You can modify the default IP address and ports for Ableton Live and TouchDesigner integration.

### Running the Project

To run the project, execute the `app.py` file:
```bash
python app.py
```

## Python Files Description

- `ableton_integration.py`: Integrates with Ableton Live using the `AbletonController` class.
- `app.py`: Main application logic, uses Open3D for visualization and MediaPipe for face and hand tracking.
- `config.py`: Configuration settings for the project.
- `image_utils.py`: Utility functions for image processing.
- `point_cloud.py`: Handles point cloud generation and effects.
- `vision.py`: Implements face and hand tracking.
- `visualizer.py`: Contains the visualization logic.

## Ableton Integration

The Ableton integration is handled by the `AbletonController` class in the `ableton_integration.py` file. This class allows you to control music playback, trigger clips, and adjust device parameters in Ableton Live using OSC messages.

### Main Application Logic

The main application logic is in the `app.py` file. It initializes the face and hand tracking processors, the Open3D visualizer, and the Ableton controller. The application captures frames from the webcam, processes them to detect face and hand landmarks, and updates the 3D visualization and Ableton Live parameters based on the detected landmarks.
