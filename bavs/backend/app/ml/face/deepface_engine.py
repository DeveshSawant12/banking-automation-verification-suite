"""
DeepFace engine wrapper — Module 4 (Face Verification).

Locked decisions:
  - Detector backend: retinaface (most accurate, used for both ID-photo
    crops and selfies — ID photos are often small/angled, selfies can have
    background clutter, both benefit from retinaface's accuracy over
    lighter detectors).
  - Recognition model: Facenet512.
  - API usage style: DeepFace.represent() for embeddings + manual cosine
    similarity (NOT DeepFace.verify()), per locked decision — this lets us
    persist raw embeddings independently (e.g. for future duplicate-
    customer detection) rather than only getting a black-box match/no-match.

Match threshold: DeepFace publishes empirically-validated cosine distance
thresholds per model, derived from benchmark evaluation (e.g. LFW). For
Facenet512, the published threshold is cosine_distance < 0.30 => match.
This is DeepFace's own calibrated value, not invented here — we import it
directly from the deepface library's verification module rather than
hardcoding a number that could drift from the library's actual default.

No silent failure paths: if RetinaFace finds zero faces in an image, this
raises a typed exception (NoFaceDetectedError) rather than returning a
zero vector or skipping verification silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from deepface import DeepFace
from deepface.modules import verification as deepface_verification

logger = logging.getLogger(__name__)

DETECTOR_BACKEND = "retinaface"
RECOGNITION_MODEL = "Facenet512"
DISTANCE_METRIC = "cosine"


class NoFaceDetectedError(Exception):
    """Raised when the configured face detector finds zero faces in an image."""


class MultipleFacesDetectedError(Exception):
    """
    Raised when more than one face is detected in an image where exactly
    one is expected (e.g. a selfie should contain exactly one person).
    Not raised for ID photos, where DeepFace.represent() with
    enforce_detection=True naturally selects the most prominent detected
    face region; ambiguity is instead surfaced to the caller via the
    face_count field so the orchestrator can decide how strict to be.
    """


@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray  # shape (512,) for Facenet512
    face_confidence: float  # detector's confidence for the selected face
    face_count: int  # total faces detected in the image
    facial_area: dict  # {x, y, w, h} of the detected face region


def extract_face_embedding(image: np.ndarray) -> FaceEmbeddingResult:
    """
    Detect a face in the given image and compute its Facenet512 embedding.

    Args:
        image: BGR numpy array (OpenCV convention) — DeepFace internally
            handles the BGR->RGB conversion it needs.

    Returns:
        FaceEmbeddingResult with the 512-dim embedding and detection metadata.

    Raises:
        NoFaceDetectedError: if retinaface finds zero faces. This is a
            deliberate, typed failure — face verification must never
            proceed with a missing/fabricated embedding.
    """
    try:
        representations = DeepFace.represent(
            img_path=image,
            model_name=RECOGNITION_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        # DeepFace raises ValueError with "Face could not be detected" when
        # enforce_detection=True and no face is found. We convert this into
        # our own typed exception so callers don't need to know DeepFace's
        # internal error message format.
        if "face" in str(exc).lower() or "detect" in str(exc).lower():
            raise NoFaceDetectedError(
                f"No face detected in image using {DETECTOR_BACKEND} backend: {exc}"
            ) from exc
        raise

    if not representations:
        raise NoFaceDetectedError(
            f"DeepFace.represent() returned no results for {DETECTOR_BACKEND} backend."
        )

    # DeepFace.represent() returns a list (one entry per detected face).
    # We take the first (most prominent / highest-confidence) result and
    # report the total count so the caller can flag images with multiple
    # faces if that matters for their use case (e.g. a selfie containing
    # more than one person).
    primary = representations[0]
    embedding = np.array(primary["embedding"], dtype=np.float32)

    if embedding.shape != (512,):
        raise RuntimeError(
            f"Unexpected Facenet512 embedding shape: {embedding.shape}, "
            f"expected (512,). This indicates a DeepFace model_name/version "
            f"mismatch."
        )

    return FaceEmbeddingResult(
        embedding=embedding,
        face_confidence=float(primary.get("face_confidence", 0.0)),
        face_count=len(representations),
        facial_area=primary.get("facial_area", {}),
    )


def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings. Returns a value in
    [-1, 1], where 1.0 means identical direction (perfect match) and
    values near 0 or negative indicate dissimilarity.
    """
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(
            f"Embedding shape mismatch: {embedding_a.shape} vs {embedding_b.shape}"
        )

    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with a zero-norm embedding.")

    return float(dot_product / (norm_a * norm_b))


def get_facenet512_match_threshold() -> float:
    """
    Return DeepFace's own published cosine-distance threshold for
    Facenet512 + cosine metric, imported directly from the deepface
    library (deepface.modules.verification.find_threshold) rather than
    hardcoded, so it stays in sync with whatever value the installed
    deepface version actually uses internally.
    """
    return float(
        deepface_verification.find_threshold(RECOGNITION_MODEL, DISTANCE_METRIC)
    )


@dataclass
class FaceMatchResult:
    cosine_similarity: float  # raw similarity, range [-1, 1]
    cosine_distance: float  # 1 - cosine_similarity, range [0, 2]
    match_percentage: float  # human-readable: (1 - distance) * 100, clipped to [0, 100]
    is_match: bool  # cosine_distance < DeepFace's calibrated Facenet512 threshold
    threshold_used: float


def verify_faces(embedding_a: np.ndarray, embedding_b: np.ndarray) -> FaceMatchResult:
    """
    Compare two face embeddings and produce a full match result using
    DeepFace's own calibrated Facenet512 threshold for the match decision.
    """
    similarity = compute_cosine_similarity(embedding_a, embedding_b)
    distance = 1.0 - similarity
    threshold = get_facenet512_match_threshold()

    match_percentage = max(0.0, min(100.0, (1.0 - distance) * 100.0))

    return FaceMatchResult(
        cosine_similarity=similarity,
        cosine_distance=distance,
        match_percentage=match_percentage,
        is_match=distance < threshold,
        threshold_used=threshold,
    )
