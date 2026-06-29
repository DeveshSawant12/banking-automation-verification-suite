"""
Synthetic Tamper Generator.

Generates labeled TAMPERED training examples from a REAL document image,
using well-documented digital image forgery techniques used throughout
forensic image-analysis literature. This is necessary because no public
real-vs-tampered Aadhaar/PAN dataset exists (confirmed project decision)
— you supply genuine sample documents, and this module produces
synthetically tampered variants so the Random Forest in
random_forest_model.py has both classes to learn from.

Techniques implemented (each is a real, named, documented forgery
technique — none invented for this project):

1. COPY-MOVE: a region of the image is copied and pasted elsewhere within
   the same image, with optional rotation. Common technique for duplicating
   a digit or altering small visual sections within a document.

2. SPLICING: a region from a DIFFERENT (donor) image is pasted into the
   target image, optionally with blending. Simulates compositing content
   from another document/source.

3. TEXT OVERLAY FORGERY: existing text region is blanked out (inpainted or
   solid-filled) and new text is rendered on top — simulates altering a
   name, DOB, or number field directly.

4. RECOMPRESSION ARTIFACT INJECTION: the image is saved through multiple
   JPEG compression cycles at varying quality levels — simulates the
   compression-history inconsistency that occurs when an image is edited
   in software and re-saved (a real, well-documented technique for
   generating recompression-artifact training signal for ELA-based
   classifiers specifically).

Each function returns the tampered image AND a tamper_metadata dict
recording exactly what was done and where, for dataset auditability
(important so you can sanity-check the generated training set is
realistic, not just "noise").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TamperMetadata:
    technique: str
    region: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    details: dict = field(default_factory=dict)


def _load_bgr(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image


def apply_copy_move(
    image: np.ndarray,
    region_size_ratio: float = 0.15,
    rotation_degrees: float | None = None,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, TamperMetadata]:
    """
    Copy a random rectangular region of the image and paste it at a
    different random location, with optional rotation. Simulates
    duplicating a section of a document (e.g. to obscure or replace a
    digit/character).
    """
    rng = rng or random.Random()
    h, w = image.shape[:2]
    region_w = max(int(w * region_size_ratio), 10)
    region_h = max(int(h * region_size_ratio), 10)

    src_x = rng.randint(0, max(w - region_w, 1) - 1) if w > region_w else 0
    src_y = rng.randint(0, max(h - region_h, 1) - 1) if h > region_h else 0

    patch = image[src_y : src_y + region_h, src_x : src_x + region_w].copy()

    if rotation_degrees is None:
        rotation_degrees = rng.choice([0, 0, 0, 5, -5, 10, -10])
    if rotation_degrees != 0:
        center = (region_w // 2, region_h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
        patch = cv2.warpAffine(patch, rot_matrix, (region_w, region_h))

    dst_x = rng.randint(0, max(w - region_w, 1) - 1) if w > region_w else 0
    dst_y = rng.randint(0, max(h - region_h, 1) - 1) if h > region_h else 0

    # Avoid pasting back onto (near) the exact same location, which would
    # produce a no-op tamper.
    attempts = 0
    while abs(dst_x - src_x) < region_w // 2 and abs(dst_y - src_y) < region_h // 2:
        dst_x = rng.randint(0, max(w - region_w, 1) - 1) if w > region_w else 0
        dst_y = rng.randint(0, max(h - region_h, 1) - 1) if h > region_h else 0
        attempts += 1
        if attempts > 10:
            break

    tampered = image.copy()
    tampered[dst_y : dst_y + region_h, dst_x : dst_x + region_w] = patch

    metadata = TamperMetadata(
        technique="copy_move",
        region=(dst_x, dst_y, region_w, region_h),
        details={
            "source_region": (src_x, src_y, region_w, region_h),
            "rotation_degrees": rotation_degrees,
        },
    )
    return tampered, metadata


def apply_splicing(
    target_image: np.ndarray,
    donor_image: np.ndarray,
    region_size_ratio: float = 0.15,
    blend: bool = True,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, TamperMetadata]:
    """
    Paste a region from a donor image into the target image. Optionally
    alpha-blends the edges to simulate a more carefully concealed splice
    (vs. a hard-edged, more obviously visible splice).

    Args:
        target_image: the document being tampered (BGR numpy array)
        donor_image: a DIFFERENT image providing the spliced-in content
            (must be a genuinely different image — using the same image as
            both target and donor degrades to copy_move, which has its own
            dedicated function above)
    """
    rng = rng or random.Random()
    th, tw = target_image.shape[:2]
    dh, dw = donor_image.shape[:2]

    region_w = max(int(tw * region_size_ratio), 10)
    region_h = max(int(th * region_size_ratio), 10)
    region_w = min(region_w, dw, tw)
    region_h = min(region_h, dh, th)

    donor_x = rng.randint(0, max(dw - region_w, 1) - 1) if dw > region_w else 0
    donor_y = rng.randint(0, max(dh - region_h, 1) - 1) if dh > region_h else 0
    patch = donor_image[donor_y : donor_y + region_h, donor_x : donor_x + region_w]

    if patch.shape[:2] != (region_h, region_w):
        patch = cv2.resize(patch, (region_w, region_h))

    dst_x = rng.randint(0, max(tw - region_w, 1) - 1) if tw > region_w else 0
    dst_y = rng.randint(0, max(th - region_h, 1) - 1) if th > region_h else 0

    tampered = target_image.copy()

    if blend:
        mask = np.zeros((region_h, region_w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (region_w // 2, region_h // 2),
            (region_w // 2, region_h // 2),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        roi = tampered[dst_y : dst_y + region_h, dst_x : dst_x + region_w].astype(
            np.float32
        )
        patch_f = patch.astype(np.float32)
        blended = roi * (1 - mask_3ch) + patch_f * mask_3ch
        tampered[dst_y : dst_y + region_h, dst_x : dst_x + region_w] = blended.astype(
            np.uint8
        )
    else:
        tampered[dst_y : dst_y + region_h, dst_x : dst_x + region_w] = patch

    metadata = TamperMetadata(
        technique="splicing",
        region=(dst_x, dst_y, region_w, region_h),
        details={"blended": blend},
    )
    return tampered, metadata


def apply_text_overlay_forgery(
    image: np.ndarray,
    replacement_text: str,
    region: tuple[int, int, int, int] | None = None,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, TamperMetadata]:
    """
    Blank out a region of the image (simulating removal of original printed
    text) and render new text on top — simulates a directly altered field
    (e.g. a changed name, DOB, or ID number).

    Args:
        region: (x, y, w, h) to overlay. If None, a random region in the
            lower-middle portion of the image is chosen (where ID
            number/name fields typically sit on Aadhaar/PAN layouts).
    """
    rng = rng or random.Random()
    h, w = image.shape[:2]

    if region is None:
        region_w = int(w * rng.uniform(0.25, 0.4))
        region_h = int(h * rng.uniform(0.05, 0.08))
        x = rng.randint(int(w * 0.1), max(int(w * 0.5), 1))
        y = rng.randint(int(h * 0.5), max(int(h * 0.85), 1))
        region = (x, y, region_w, region_h)

    x, y, region_w, region_h = region

    tampered = image.copy()
    # Sample the surrounding background color to blank the region
    # plausibly rather than with a flat unrelated color.
    border_sample = tampered[max(y - 2, 0) : y, x : x + region_w]
    if border_sample.size > 0:
        fill_color = tuple(int(c) for c in np.mean(border_sample, axis=(0, 1)))
    else:
        fill_color = (255, 255, 255)

    cv2.rectangle(
        tampered, (x, y), (x + region_w, y + region_h), fill_color, thickness=-1
    )

    # Render new text using PIL (better font rendering control than cv2.putText)
    pil_img = Image.fromarray(cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(region_h - 6, 10))
    except OSError:
        font = ImageFont.load_default()
    draw.text((x + 2, y + 2), replacement_text, fill=(20, 20, 20), font=font)

    tampered = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    metadata = TamperMetadata(
        technique="text_overlay_forgery",
        region=region,
        details={"replacement_text": replacement_text},
    )
    return tampered, metadata


def apply_recompression_artifacts(
    image: np.ndarray, n_cycles: int = 3, quality_range: tuple[int, int] = (50, 85)
) -> tuple[np.ndarray, TamperMetadata]:
    """
    Re-save the image through multiple JPEG compression cycles at varying
    quality levels. Simulates the compression-history inconsistency
    produced when a document is edited in image software and re-exported,
    which is specifically what ELA is designed to detect — used to ensure
    the Random Forest sees genuine recompression-artifact patterns, not
    just spatial copy-move/splice signal.
    """
    rng = random.Random()
    qualities = [rng.randint(*quality_range) for _ in range(n_cycles)]

    current = image
    for quality in qualities:
        success, encoded = cv2.imencode(".jpg", current, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("JPEG encoding failed during recompression simulation.")
        current = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    metadata = TamperMetadata(
        technique="recompression_artifacts",
        details={"n_cycles": n_cycles, "qualities": qualities},
    )
    return current, metadata


TAMPER_TECHNIQUES = (
    "copy_move",
    "splicing",
    "text_overlay_forgery",
    "recompression_artifacts",
)


def generate_random_tamper(
    image: np.ndarray,
    donor_image: np.ndarray | None = None,
    replacement_texts: list[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, TamperMetadata]:
    """
    Apply ONE randomly selected tampering technique to the given image.
    Used by the dataset-building script to generate diverse tampered
    examples from a set of real source documents.

    Args:
        image: BGR numpy array of a REAL (untampered) source document
        donor_image: required if 'splicing' is selected; if None, splicing
            is excluded from the random choice for this call
        replacement_texts: candidate strings for text_overlay_forgery;
            if None, a generic placeholder digit string is used
    """
    rng = rng or random.Random()

    available_techniques = list(TAMPER_TECHNIQUES)
    if donor_image is None:
        available_techniques.remove("splicing")

    technique = rng.choice(available_techniques)

    if technique == "copy_move":
        return apply_copy_move(image, rng=rng)
    elif technique == "splicing":
        return apply_splicing(image, donor_image, rng=rng)
    elif technique == "text_overlay_forgery":
        texts = replacement_texts or ["1234 5678 9012", "ABCDE1234F", "01/01/2000"]
        return apply_text_overlay_forgery(image, rng.choice(texts), rng=rng)
    elif technique == "recompression_artifacts":
        return apply_recompression_artifacts(image)
    else:
        raise ValueError(f"Unknown technique: {technique}")
