#!/usr/bin/env python3
"""
CLI training script for the Aadhaar/PAN tampering Random Forest classifier.

USAGE:
    python -m app.ml.forgery.train_forgery_model \\
        --real-dir training_data/raw/aadhaar \\
        --output ml_models/aadhaar_rf_model.pkl \\
        --tampered-per-real 3

This script:
  1. Loads REAL document images from --real-dir (you supply these — must
     be genuine Aadhaar or PAN images, per the locked project decision
     that no synthetic/fabricated "real" documents will be invented).
  2. For each real image, generates N synthetically tampered variants
     using synthetic_tamper_generator.py (random technique selection).
  3. Runs the full feature extraction pipeline (OCR -> ELA -> ResNet18 ->
     fusion) on every real and tampered image.
  4. Trains a ForgeryRandomForestModel on the resulting labeled dataset.
  5. Saves the trained model + evaluation report to --output.

This script requires the same dependencies as the live inference service
(EasyOCR, OpenCV, torch/torchvision) and will download EasyOCR/ResNet18
model weights on first run if not already cached — run this in your
Docker/local environment, not in a network-restricted sandbox.

REQUIRES AT LEAST 2 real source images to produce a stratified train/test
split with both classes represented; in practice, dozens-to-hundreds of
real source images (each producing --tampered-per-real synthetic
variants) are needed for the Random Forest to generalize meaningfully.
This script will run and produce a model with very few images, but will
print a prominent warning if the dataset is too small to be reliable.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.ml.forgery.ela import run_ela_pipeline
from app.ml.forgery.feature_fusion import fuse_features
from app.ml.forgery.ocr_derived_features import extract_ocr_derived_features
from app.ml.forgery.random_forest_model import (
    LABEL_REAL,
    LABEL_TAMPERED,
    ForgeryRandomForestModel,
)
from app.ml.forgery.resnet_feature_extractor import extract_resnet_features
from app.ml.forgery.synthetic_tamper_generator import generate_random_tamper
from app.ml.ocr.easyocr_engine import run_ocr
from app.ml.ocr.field_parser import parse_fields
from app.utils.image_utils import ImageLoadError, preprocess_for_ocr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_RECOMMENDED_REAL_IMAGES = 20


def extract_full_feature_vector(image_path: Path) -> np.ndarray:
    """
    Run the complete OCR -> ELA -> ResNet18 -> fusion pipeline on a single
    image file and return its fused feature vector. Shared by both the
    real-image and tampered-image paths during dataset construction.
    """
    file_bytes = image_path.read_bytes()

    try:
        ocr_input_image = preprocess_for_ocr(file_bytes)
    except ImageLoadError as exc:
        raise RuntimeError(f"Failed to load {image_path} for OCR: {exc}") from exc

    ocr_results = run_ocr(ocr_input_image)
    parsed_fields = parse_fields(ocr_results)
    ocr_features = extract_ocr_derived_features(ocr_results, parsed_fields)

    pil_image = Image.open(image_path).convert("RGB")
    _, ela_features = run_ela_pipeline(pil_image)
    resnet_features = extract_resnet_features(pil_image)

    return fuse_features(ela_features, resnet_features, ocr_features)


def build_dataset(
    real_dir: Path, tampered_per_real: int, donor_pool: list[Path]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build the labeled (X, y) dataset from a directory of real images.

    Returns:
        X: (n_samples, n_features) fused feature matrix
        y: (n_samples,) labels (LABEL_REAL / LABEL_TAMPERED)
        sample_sources: list of strings describing each sample's origin,
            for the training metadata log (auditability)
    """
    real_image_paths = sorted(
        [
            p
            for p in real_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )

    if len(real_image_paths) == 0:
        raise ValueError(
            f"No real images found in {real_dir}. Supply genuine Aadhaar/PAN "
            f"sample images before running training."
        )

    if len(real_image_paths) < MIN_RECOMMENDED_REAL_IMAGES:
        logger.warning(
            "Only %d real images found (recommended minimum: %d). The "
            "trained model is unlikely to generalize well. This will "
            "proceed, but treat the resulting model as a development "
            "prototype, not production-ready, until more data is added.",
            len(real_image_paths),
            MIN_RECOMMENDED_REAL_IMAGES,
        )

    rng = random.Random(42)
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    sources: list[str] = []

    for real_path in real_image_paths:
        logger.info("Processing REAL image: %s", real_path.name)
        try:
            real_vector = extract_full_feature_vector(real_path)
        except Exception as exc:
            logger.error("Skipping %s due to extraction failure: %s", real_path, exc)
            continue

        features_list.append(real_vector)
        labels_list.append(LABEL_REAL)
        sources.append(f"REAL:{real_path.name}")

        real_bgr = cv2.imread(str(real_path))
        for i in range(tampered_per_real):
            donor_bgr = None
            if donor_pool:
                donor_path = rng.choice(donor_pool)
                donor_bgr = cv2.imread(str(donor_path))

            try:
                tampered_bgr, tamper_meta = generate_random_tamper(
                    real_bgr, donor_image=donor_bgr, rng=rng
                )
                tampered_pil = Image.fromarray(
                    cv2.cvtColor(tampered_bgr, cv2.COLOR_BGR2RGB)
                )

                tmp_path = real_path.parent / f".tmp_tampered_{real_path.stem}_{i}.jpg"
                tampered_pil.save(tmp_path, "JPEG", quality=92)

                tampered_vector = extract_full_feature_vector(tmp_path)
                tmp_path.unlink(missing_ok=True)

            except Exception as exc:
                logger.error(
                    "Skipping tampered variant %d of %s due to error: %s",
                    i,
                    real_path.name,
                    exc,
                )
                continue

            features_list.append(tampered_vector)
            labels_list.append(LABEL_TAMPERED)
            sources.append(
                f"TAMPERED:{real_path.name}:variant_{i}:{tamper_meta.technique}"
            )

    X = np.stack(features_list)
    y = np.array(labels_list)
    return X, y, sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Aadhaar/PAN tampering detection Random Forest."
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Directory containing genuine (untampered) document images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the trained model (e.g. ml_models/aadhaar_rf_model.pkl)",
    )
    parser.add_argument(
        "--tampered-per-real",
        type=int,
        default=3,
        help="Number of synthetic tampered variants to generate per real image.",
    )
    parser.add_argument(
        "--donor-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of additional images used as splicing donors. "
            "If omitted, splicing is excluded from the random technique pool."
        ),
    )
    args = parser.parse_args()

    if not args.real_dir.exists():
        logger.error("--real-dir does not exist: %s", args.real_dir)
        sys.exit(1)

    donor_pool: list[Path] = []
    if args.donor_dir is not None:
        if not args.donor_dir.exists():
            logger.error("--donor-dir does not exist: %s", args.donor_dir)
            sys.exit(1)
        donor_pool = sorted(
            [
                p
                for p in args.donor_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )

    logger.info("Building dataset...")
    X, y, sources = build_dataset(args.real_dir, args.tampered_per_real, donor_pool)
    logger.info(
        "Dataset built: %d samples (%d REAL, %d TAMPERED), %d features.",
        X.shape[0],
        int(np.sum(y == LABEL_REAL)),
        int(np.sum(y == LABEL_TAMPERED)),
        X.shape[1],
    )

    model = ForgeryRandomForestModel()
    report = model.train(X, y)

    logger.info("Training complete. Evaluation report:")
    logger.info("  Accuracy:  %.4f", report["accuracy"])
    logger.info("  Precision: %.4f", report["precision"])
    logger.info("  Recall:    %.4f", report["recall"])
    logger.info("  F1 Score:  %.4f", report["f1_score"])
    logger.info("  Confusion Matrix: %s", report["confusion_matrix"])

    model.save(args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
