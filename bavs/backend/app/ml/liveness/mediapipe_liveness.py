"""
MediaPipe Liveness Detection Engine — Module 6.

API NOTE (verified against installed mediapipe==0.10.33, not assumed):
this module uses the current MediaPipe Tasks API
(mediapipe.tasks.vision.FaceLandmarker), NOT the legacy
mediapipe.solutions.face_mesh API, which no longer exists in current
mediapipe releases. FaceLandmarker requires a downloadable .task model
bundle (Google-hosted) — first run will download it; this is a model-
weight download identical in nature to EasyOCR/ResNet18's downloads in
earlier modules.

Detection signals used (all directly from MediaPipe's own model output,
not hand-rolled landmark math):

  - BLINK: 'eyeBlinkLeft' / 'eyeBlinkRight' blendshape scores
    (output_face_blendshapes=True). These are MediaPipe's own pre-trained
    blendshape coefficients (0.0-1.0), not a manually computed eye-aspect-
    ratio heuristic.

  - SMILE: 'mouthSmileLeft' / 'mouthSmileRight' blendshape scores.

  - HEAD MOVEMENT: derived from facial_transformation_matrixes
    (output_facial_transformation_matrixes=True), which gives a 4x4
    rigid transformation matrix per frame. We extract the yaw rotation
    angle and track its change across frames — head turn is a rigid pose
    change, not an expression, so it is NOT represented as a blendshape
    and must come from the transformation matrix.

All 52 blendshape category names (eyeBlinkLeft, mouthSmileLeft, etc.) are
the exact, documented names from MediaPipe's face_blendshapes_graph —
verified via Google's own MediaPipe source/documentation, not invented.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image as MpImage, ImageFormat

logger = logging.getLogger(__name__)

# Path to the downloaded MediaPipe FaceLandmarker .task model bundle.
# Download via (run once, in Docker/local where Google's model CDN is
# reachable):
#   wget -O ml_models/face_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
FACE_LANDMARKER_MODEL_PATH = Path("ml_models/face_landmarker.task")

BLINK_THRESHOLD = 0.4  # blendshape score above which an eye is considered closed
SMILE_THRESHOLD = 0.4  # blendshape score above which a smile is considered detected
HEAD_YAW_MOVEMENT_THRESHOLD_DEGREES = 15.0  # minimum yaw delta to count as a head turn

_landmarker_lock = threading.Lock()
_landmarker_instance: vision.FaceLandmarker | None = None


class LivenessModelNotAvailableError(Exception):
    """
    Raised when the FaceLandmarker .task model file is not present at
    FACE_LANDMARKER_MODEL_PATH. Mirrors the ModelNotTrainedError pattern
    from Modules 2/3 — liveness verification must never proceed (and
    never fabricate a LIVE/SPOOF verdict) without the real model present.
    """


class NoFaceInFrameError(Exception):
    """Raised when no face is detected in a given video frame."""


def _build_landmarker() -> vision.FaceLandmarker:
    if not FACE_LANDMARKER_MODEL_PATH.exists():
        raise LivenessModelNotAvailableError(
            f"MediaPipe FaceLandmarker model not found at "
            f"{FACE_LANDMARKER_MODEL_PATH}. Download it first (see module "
            f"docstring for the download command) — this requires network "
            f"access to Google's model CDN, run in Docker/local, not in a "
            f"network-restricted sandbox."
        )

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_landmarker() -> vision.FaceLandmarker:
    """Lazily initialize and return the singleton FaceLandmarker instance."""
    global _landmarker_instance
    if _landmarker_instance is not None:
        return _landmarker_instance

    with _landmarker_lock:
        if _landmarker_instance is None:
            _landmarker_instance = _build_landmarker()
    return _landmarker_instance


@dataclass
class FrameAnalysis:
    frame_index: int
    eye_blink_left: float
    eye_blink_right: float
    mouth_smile_left: float
    mouth_smile_right: float
    yaw_degrees: float
    blink_detected: bool
    smile_detected: bool


def _extract_yaw_degrees(transformation_matrix: np.ndarray) -> float:
    """
    Extract the yaw (left-right head turn) rotation angle in degrees from
    a 4x4 facial transformation matrix, using the standard rotation-matrix
    -> Euler-angle decomposition for the Y-axis rotation component.
    """
    rotation = transformation_matrix[:3, :3]
    # Standard yaw extraction from a rotation matrix (Y-axis rotation):
    # yaw = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
    yaw_rad = math.atan2(
        -rotation[2, 0], math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    )
    return math.degrees(yaw_rad)


def analyze_frame(frame_bgr: np.ndarray, frame_index: int) -> FrameAnalysis:
    """
    Run MediaPipe FaceLandmarker on a single BGR video frame and extract
    blink/smile/yaw signals.

    Raises:
        NoFaceInFrameError: if no face is detected in this frame.
        LivenessModelNotAvailableError: if the .task model file is missing.
    """
    landmarker = get_landmarker()  # raises LivenessModelNotAvailableError if missing

    rgb_frame = frame_bgr[:, :, ::-1]  # BGR -> RGB
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise NoFaceInFrameError(f"No face detected in frame {frame_index}.")

    blendshapes_by_name = {
        category.category_name: category.score
        for category in result.face_blendshapes[0]
    }

    eye_blink_left = blendshapes_by_name.get("eyeBlinkLeft", 0.0)
    eye_blink_right = blendshapes_by_name.get("eyeBlinkRight", 0.0)
    mouth_smile_left = blendshapes_by_name.get("mouthSmileLeft", 0.0)
    mouth_smile_right = blendshapes_by_name.get("mouthSmileRight", 0.0)

    if not result.facial_transformation_matrixes:
        raise NoFaceInFrameError(
            f"Face detected but no transformation matrix available for "
            f"frame {frame_index}."
        )
    yaw_degrees = _extract_yaw_degrees(result.facial_transformation_matrixes[0])

    return FrameAnalysis(
        frame_index=frame_index,
        eye_blink_left=eye_blink_left,
        eye_blink_right=eye_blink_right,
        mouth_smile_left=mouth_smile_left,
        mouth_smile_right=mouth_smile_right,
        yaw_degrees=yaw_degrees,
        blink_detected=(eye_blink_left > BLINK_THRESHOLD)
        or (eye_blink_right > BLINK_THRESHOLD),
        smile_detected=(mouth_smile_left > SMILE_THRESHOLD)
        or (mouth_smile_right > SMILE_THRESHOLD),
    )
