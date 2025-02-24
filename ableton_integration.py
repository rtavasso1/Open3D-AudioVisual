import time
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple, Union, Any, Dict, Callable, Optional
from pythonosc import udp_client

class ClipState(Enum):
    STOPPED = 0
    PLAYING = 1
    TRIGGERED = 2

class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

@dataclass
class Clip:
    name: str
    index: int
    state: ClipState = ClipState.STOPPED
    last_triggered: float = 0

@dataclass
class DeviceParameter:
    name: str
    index: int
    min_val: float = 0.0
    max_val: float = 127.0
    mapping_fn: Optional[Callable[[float], float]] = None
    current_value: float = 0.0
    smoothing_enabled: bool = False
    smoothing_method: str = "moving_average"
    smoothing_window_size: int = 5
    alpha: float = 0.2
    _value_history: deque = field(default_factory=lambda: deque(maxlen=20), init=False)

    def normalize_value(self, value: float) -> float:
        """Maps an incoming 0..1 to min_val..max_val, with optional custom mapping."""
        assert 0.0 <= value <= 1.0, f"Input must be between 0 and 1, got {value}"
        if self.mapping_fn:
            value = self.mapping_fn(value)
        return self.min_val + value * (self.max_val - self.min_val)

    def update_value(self, raw_value: float) -> float:
        normalized = self.normalize_value(raw_value)
        if not self.smoothing_enabled:
            self.current_value = normalized
            return self.current_value

        # If smoothing is enabled, choose the method
        if self.smoothing_method == "moving_average":
            self._value_history.append(normalized)
            self.current_value = sum(self._value_history) / len(self._value_history)
        elif self.smoothing_method == "exponential":
            self.current_value += self.alpha * (normalized - self.current_value)
        else:
            self.current_value = normalized
        return self.current_value

@dataclass
class Device:
    name: str
    index: int
    parameters: Dict[str, DeviceParameter]

@dataclass
class Track:
    name: str
    index: int
    devices: Dict[str, Device]
    clips: List[Clip]
    active_clip: Optional[int] = None

class ParameterControl:
    def __init__(self, track: Track, device: Device, parameter: DeviceParameter):
        self.track = track
        self.device = device
        self.parameter = parameter

class PoseControl:
    """Generic class for controlling a parameter from a certain hand pose measurement."""
    def __init__(self, name: str,
                 parameter_control: ParameterControl,
                 landmark_indices: List[int],
                 calculation_method: Callable[[List[Any]], float],
                 sensitivity: float = 1.0):
        self.name = name
        self.parameter_control = parameter_control
        self.landmark_indices = landmark_indices
        self.calculation_method = calculation_method
        self.sensitivity = sensitivity

    def calculate_value(self, hand_landmarks) -> float:
        if not hand_landmarks:
            return 0.0
        subset = [hand_landmarks.landmark[idx] for idx in self.landmark_indices]
        raw_value = self.calculation_method(subset)
        adjusted_value = pow(raw_value, 1 / self.sensitivity)
        return min(1.0, max(0.0, adjusted_value))

class HandCalculations:
    """Example methods for reading numeric values from hand landmark subsets."""
    @staticmethod
    def thumb_index_distance(landmarks) -> float:
        thumb_tip = landmarks[0]
        index_tip = landmarks[1]
        dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        return min(dist * 3, 1.0)

    @staticmethod
    def vertical_hand_position(landmarks) -> float:
        # Example: if you wanted wrist Y -> 0..1
        wrist = landmarks[0]
        return min(max(1.0 - wrist.y, 0.0), 1.0)

class AbletonController:
    def __init__(self, ip: str, port: int):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.tracks = self._setup_tracks()
        self.current_scene = 0
        self.last_message_times = {}
        self.message_interval = 1.0 / 5.0  # rate-limit parameter messages

    def _setup_tracks(self) -> Dict[str, Track]:
        """Define the structure of your Ableton set's tracks, devices, and clips."""

        def stepped(x: float, steps: int = 12) -> float:
            """Example mapping function to snap to discrete steps."""
            return round(x * steps) / steps

        return {
            "Bass": Track(
                name="Bass",
                index=0,
                devices={
                    "distort_rack": Device(
                        name="Audio Rack",
                        index=0,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                    "space_rack": Device(
                        name="Audio Rack",
                        index=1,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                },
                clips=[
                    Clip(name="Full Beat", index=4),
                ]
            ),
            "drums": Track(
                name="drums",
                index=1,
                devices={
                    "distort_rack": Device(
                        name="Audio Rack",
                        index=0,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                    "space_rack": Device(
                        name="Audio Rack",
                        index=1,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                },
                clips=[
                    Clip(name="Lead Melody", index=0),
                ]
            ),
            "other": Track(
                name="other",
                index=2,
                devices={
                    "distort_rack": Device(
                        name="Audio Rack",
                        index=0,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                    "space_rack": Device(
                        name="Audio Rack",
                        index=1,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                },
                clips=[
                    Clip(name="Ambient Pad", index=0)
                ]
            ),
            "vocal": Track(
                name="vocal",
                index=3,
                devices={
                    "distort_rack": Device(
                        name="Audio Rack",
                        index=0,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                    "space_rack": Device(
                        name="Audio Rack",
                        index=1,
                        parameters={
                            "macro1": DeviceParameter(name="Macro 1", index=1, min_val=0, max_val=127, mapping_fn=stepped),
                        }
                    ),
                },
                clips=[
                    Clip(name="Ambient Pad", index=0)
                ]
            )
        }

    #
    # --- Playback control methods ---
    #

    def start_song_playback(self):
        self.client.send_message("/live/song/start_playing", [])

    def stop_song_playback(self):
        self.client.send_message("/live/song/stop_playing", [])

    def play_clip(self, track_idx: int, clip_idx: int) -> None:
        """Fires a specific clip in a track."""
        self.client.send_message("/live/clip/fire", [track_idx, clip_idx])
        # Update internal track states
        for track in self.tracks.values():
            if track.index == track_idx:
                track.active_clip = clip_idx
                for clip in track.clips:
                    if clip.index == clip_idx:
                        clip.state = ClipState.PLAYING
                        clip.last_triggered = time.time()

    def stop_clip(self, track_idx: int, clip_idx: int) -> None:
        """Stops a specific clip in a track."""
        self.client.send_message("/live/clip/stop", [track_idx, clip_idx])
        for track in self.tracks.values():
            if track.index == track_idx:
                for clip in track.clips:
                    if clip.index == clip_idx:
                        clip.state = ClipState.STOPPED
                if track.active_clip == clip_idx:
                    track.active_clip = None

    def stop_track(self, track_idx: int) -> None:
        """Stops all clips in a single track."""
        self.client.send_message("/live/track/stop_all_clips", [track_idx])
        for track in self.tracks.values():
            if track.index == track_idx:
                for clip in track.clips:
                    clip.state = ClipState.STOPPED
                track.active_clip = None

    #
    # --- Track-level controls ---
    #

    def set_track_mute(self, track_name: str, mute_state: int) -> None:
        """
        Mutes or unmutes a track.
        mute_state = 1 -> Mute, 0 -> Unmute
        """
        track = self.tracks.get(track_name)
        if track is not None:
            self.client.send_message("/live/track/set/mute", [track.index, mute_state])

    def set_tempo(self, tempo: float) -> None:
        """Set the global BPM."""
        self.client.send_message("/live/song/set/tempo", [float(tempo)])

    #
    # --- Parameter control (macro) ---
    #

    def set_parameter(self, control: 'ParameterControl', value: float) -> None:
        """
        Sets a device parameter value, with rate-limiting to avoid spamming OSC.
        """
        key = (control.track.index, control.device.index, control.parameter.index)
        current_time = time.time()
        last_time = self.last_message_times.get(key, 0)
        if current_time - last_time < self.message_interval:
            return
        self.last_message_times[key] = current_time

        smoothed_value = control.parameter.update_value(value)
        self.client.send_message(
            "/live/device/set/parameter/value",
            [control.track.index, control.device.index, control.parameter.index, smoothed_value]
        )

    def update_rack_macro(self, rack_key: str, value: float) -> None:
        """
        Convenience method to set the 'macro1' parameter on a rack across all tracks, if present.
        Example: controlling a global 'space_rack' or 'distort_rack' on each track.
        """
        # Print for debug; remove if not needed
        print(f"Updating rack {rack_key} with value {value}")

        for track in self.tracks.values():
            if rack_key in track.devices:
                device = track.devices[rack_key]
                param = device.parameters["macro1"]
                smoothed_value = param.update_value(value)
                self.client.send_message(
                    "/live/device/set/parameter/value",
                    [track.index, device.index, param.index, smoothed_value]
                )
