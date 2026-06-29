"""
Random Forest tampering classifier — train/predict/save/load lifecycle.

IMPORTANT — this is a fully functional, executable module, but it has NO
trained model artifact shipped with it, because per the locked project
decision, no public real-vs-tampered Aadhaar/PAN dataset exists and none
will be fabricated. You (the project owner) must run
train_forgery_model.py against labeled data (real docs + synthetically
tampered versions produced by synthetic_tamper_generator.py) to produce
the .pkl artifact referenced in app/config or ml_models/.

Inference code in this module explicitly refuses to fabricate a verdict
when no trained model is present — it raises ModelNotTrainedError, which
the orchestrator must catch and route the case to REVIEW_REQUIRED. This
is a deliberate safety property, not an oversight: a banking fraud system
must never silently guess.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.ml.forgery.feature_fusion import fused_feature_names

logger = logging.getLogger(__name__)

LABEL_REAL = 0
LABEL_TAMPERED = 1
LABEL_NAMES = {LABEL_REAL: "REAL", LABEL_TAMPERED: "TAMPERED"}


class ModelNotTrainedError(Exception):
    """
    Raised when a prediction is requested but no trained Random Forest
    model artifact exists at the configured path. The caller MUST treat
    this as "verification could not be completed" (-> REVIEW_REQUIRED),
    never as an implicit REAL or TAMPERED verdict.
    """


@dataclass
class TamperPrediction:
    verdict: str  # "REAL" | "TAMPERED"
    confidence: float  # predict_proba of the predicted class
    real_probability: float
    tampered_probability: float
    model_version: str


class ForgeryRandomForestModel:
    """
    Wraps a scikit-learn RandomForestClassifier with a fixed feature
    contract (must match feature_fusion.fused_feature_names() exactly),
    versioning, and a save/load format that stores both the model and its
    training metadata together for auditability.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 20,
        min_samples_leaf: int = 2,
        class_weight: str | dict = "balanced",
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = fused_feature_names()
        self.model_version: str | None = None
        self.training_metadata: dict | None = None
        self._is_fitted = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Train the Random Forest on a fused-feature dataset and return an
        evaluation report computed on a held-out test split.

        Args:
            X: shape (n_samples, n_features) — must match
               feature_fusion.fused_vector_dimensionality()
            y: shape (n_samples,) — binary labels, LABEL_REAL or
               LABEL_TAMPERED

        Returns:
            dict containing accuracy, precision, recall, f1, confusion
            matrix, and per-class report — this is what gets written to
            the training metadata log for audit/reproducibility.

        Raises:
            ValueError: on dimensionality mismatch or insufficient class
                representation (refuses to silently train a broken model).
        """
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Training data has {X.shape[1]} features, expected "
                f"{len(self.feature_names)} (feature_fusion contract). "
                f"This indicates a mismatch between training and inference "
                f"feature extraction — refusing to train."
            )

        unique_labels = set(np.unique(y).tolist())
        if not unique_labels.issubset({LABEL_REAL, LABEL_TAMPERED}):
            raise ValueError(
                f"Labels must be {{{LABEL_REAL}, {LABEL_TAMPERED}}}, "
                f"got {unique_labels}."
            )
        if len(unique_labels) < 2:
            raise ValueError(
                "Training data must contain both REAL and TAMPERED examples. "
                f"Only found label(s): {unique_labels}."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        y_pred = self.model.predict(X_test)

        report = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["REAL", "TAMPERED"], zero_division=0
            ),
            "n_train_samples": int(X_train.shape[0]),
            "n_test_samples": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        self.model_version = f"rf_v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.training_metadata = report

        logger.info(
            "Random Forest trained. Accuracy=%.4f, F1=%.4f, version=%s",
            report["accuracy"],
            report["f1_score"],
            self.model_version,
        )
        return report

    def predict(self, feature_vector: np.ndarray) -> TamperPrediction:
        """
        Predict REAL/TAMPERED for a single fused feature vector.

        Raises:
            ModelNotTrainedError: if this model instance has not been
                trained or loaded from disk.
            ValueError: on feature dimensionality mismatch.
        """
        if not self._is_fitted:
            raise ModelNotTrainedError(
                "ForgeryRandomForestModel.predict() called on an untrained "
                "model. Train via train_forgery_model.py or load a saved "
                "model with .load() before calling predict()."
            )

        if feature_vector.shape != (len(self.feature_names),):
            raise ValueError(
                f"Expected feature vector of shape ({len(self.feature_names)},), "
                f"got {feature_vector.shape}."
            )

        probabilities = self.model.predict_proba(feature_vector.reshape(1, -1))[0]
        predicted_label = int(np.argmax(probabilities))

        return TamperPrediction(
            verdict=LABEL_NAMES[predicted_label],
            confidence=float(probabilities[predicted_label]),
            real_probability=float(probabilities[LABEL_REAL]),
            tampered_probability=float(probabilities[LABEL_TAMPERED]),
            model_version=self.model_version or "unknown",
        )

    def feature_importances(self) -> dict[str, float]:
        """
        Return feature name -> importance score mapping, sorted descending.
        Used by the Explainable AI module (Module 8) to report which
        feature groups drove a TAMPERED verdict.
        """
        if not self._is_fitted:
            raise ModelNotTrainedError(
                "Cannot compute feature importances on an untrained model."
            )
        importances = self.model.feature_importances_
        pairs = sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda p: p[1],
            reverse=True,
        )
        return dict(pairs)

    def save(self, path: str | Path) -> None:
        """
        Persist the trained model + metadata to disk. Saves two files:
        `{path}` (joblib-serialized sklearn model + feature contract) and
        `{path}.meta.json` (human-readable training metadata for audit).
        """
        if not self._is_fitted:
            raise ModelNotTrainedError("Cannot save an untrained model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "model_version": self.model_version,
            },
            path,
        )

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2)

        logger.info("Model saved to %s (metadata: %s)", path, meta_path)

    @classmethod
    def load(cls, path: str | Path) -> "ForgeryRandomForestModel":
        """
        Load a previously trained model from disk.

        Raises:
            ModelNotTrainedError: if the file does not exist at the given
                path — this is the explicit, typed signal the inference
                service layer must catch and convert into REVIEW_REQUIRED.
        """
        path = Path(path)
        if not path.exists():
            raise ModelNotTrainedError(
                f"No trained model found at {path}. Train a model first "
                f"using train_forgery_model.py. Refusing to fabricate "
                f"a verdict without a real trained classifier."
            )

        payload = joblib.load(path)

        instance = cls()
        instance.model = payload["model"]
        saved_feature_names = payload["feature_names"]
        instance.model_version = payload["model_version"]
        instance._is_fitted = True

        if saved_feature_names != instance.feature_names:
            raise ValueError(
                "Loaded model's feature contract does not match the current "
                "feature_fusion.fused_feature_names(). The model was likely "
                "trained against a different code version. Refusing to load "
                "to avoid silently corrupt predictions. Retrain the model."
            )

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                instance.training_metadata = json.load(f)

        logger.info("Model loaded from %s (version=%s)", path, instance.model_version)
        return instance
