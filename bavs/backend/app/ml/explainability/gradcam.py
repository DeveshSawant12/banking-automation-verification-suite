"""
Grad-CAM on the frozen ResNet18 feature extractor.

HONEST SCOPE STATEMENT (read before using this module): this is genuine
Grad-CAM (Selvaraju et al., 2017) — real gradient backpropagation through
ResNet18's final convolutional layer, weighted by the gradient of a
target class score w.r.t. that layer's activations. It is NOT an
explanation of the Random Forest's REAL/TAMPERED decision, because the
Random Forest is not a differentiable CNN and Grad-CAM's algorithm
fundamentally requires backprop through conv layers to a classification
score, which the RF does not have.

What this DOES explain: which spatial regions of the document image most
strongly activated ResNet18's learned ImageNet visual features (using the
network's own top-1 predicted ImageNet class as the backprop target,
since ResNet18 here has no fine-tuned tamper-specific head to target
instead). This is useful as a SUPPLEMENTARY visual signal — regions with
unusual texture/object-like activations sometimes coincide with edited
regions, since splicing/copy-move can introduce visual elements ResNet18
finds statistically unusual relative to natural document texture — but it
is not the verdict's primary explanation. That role belongs to
ela_feature_explainer.py, which explains the Random Forest directly via
feature-importance attribution over inputs it actually computed its
decision from.

Implementation note: ResNet18's frozen feature extractor (from
resnet_feature_extractor.py) has its final FC layer already stripped, so
we cannot get classification logits from it directly. This module loads
its OWN unmodified copy of pretrained ResNet18 (with the FC layer intact)
purely for Grad-CAM purposes, since Grad-CAM needs a real class score
target to backpropagate from. This is a deliberate, documented departure
from reusing resnet_feature_extractor.get_feature_extractor() — reusing
that singleton would not work for Grad-CAM (no classification head to
target), and modifying that shared singleton to add a head back would
violate "never rewrite unrelated files" by changing Module 2's tested
contract.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)

_gradcam_model_lock = threading.Lock()
_gradcam_model: torch.nn.Module | None = None
_gradcam_preprocess = None

# ResNet18's final convolutional block, by torchvision's module naming.
# This is the standard, documented target layer for Grad-CAM on ResNet
# architectures (last conv layer before global average pooling).
TARGET_LAYER_NAME = "layer4"


def _get_gradcam_model() -> tuple[torch.nn.Module, object]:
    """
    Lazily load a full (FC-layer-intact) ImageNet-pretrained ResNet18,
    separate from the feature-extractor singleton in
    resnet_feature_extractor.py, since Grad-CAM needs real classification
    logits to backpropagate from. Gradients ARE required here (unlike the
    frozen feature extractor), so parameters are NOT set to
    requires_grad=False.
    """
    global _gradcam_model, _gradcam_preprocess

    if _gradcam_model is not None:
        return _gradcam_model, _gradcam_preprocess

    with _gradcam_model_lock:
        if _gradcam_model is None:
            logger.info("Loading full ResNet18 (with FC head) for Grad-CAM...")
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            model.eval()
            _gradcam_model = model
            _gradcam_preprocess = weights.transforms()
            logger.info("Grad-CAM ResNet18 loaded.")

    return _gradcam_model, _gradcam_preprocess


class GradCamHook:
    """Captures the target layer's forward activations and backward gradients."""

    def __init__(self, target_layer: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def generate_gradcam_heatmap(pil_image: Image.Image) -> tuple[np.ndarray, int, float]:
    """
    Run Grad-CAM on the given image using ResNet18's top-1 predicted
    ImageNet class as the backprop target.

    Returns:
        (heatmap, predicted_class_idx, predicted_class_confidence)
        heatmap: np.ndarray of shape (H, W), float32, values in [0, 1],
            where H, W match the model's input resolution (224x224 — the
            caller is responsible for resizing back to the original image
            dimensions if overlaying on the full-size document).
        predicted_class_idx: ImageNet class index used as the Grad-CAM target
        predicted_class_confidence: softmax probability of that class

    Raises:
        RuntimeError: if gradient capture fails (e.g. hook didn't fire),
            which would indicate a genuine implementation problem rather
            than something to silently paper over with a blank heatmap.
    """
    model, preprocess = _get_gradcam_model()

    target_layer = dict(model.named_modules())[TARGET_LAYER_NAME]
    hook = GradCamHook(target_layer)

    try:
        input_tensor = preprocess(pil_image).unsqueeze(0)
        # requires_grad_(True) here is solely to satisfy PyTorch's
        # full_backward_hook firing condition cleanly (avoids a benign but
        # noisy UserWarning); we still only read gradients w.r.t. the
        # target conv layer's activations (captured by the hook), never
        # the input pixels themselves.
        input_tensor.requires_grad_(True)

        logits = model(input_tensor)  # (1, 1000)
        probabilities = F.softmax(logits, dim=1)

        predicted_class_idx = int(torch.argmax(logits, dim=1).item())
        predicted_class_confidence = float(probabilities[0, predicted_class_idx].item())

        model.zero_grad()
        class_score = logits[0, predicted_class_idx]
        class_score.backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError(
                "Grad-CAM hook did not capture activations/gradients. "
                "This indicates the target layer name or hook registration "
                "is broken, not a normal runtime condition to suppress."
            )

        activations = hook.activations[0]  # (C, H, W)
        gradients = hook.gradients[0]  # (C, H, W)

        # Global-average-pool the gradients over spatial dims to get
        # per-channel importance weights (standard Grad-CAM weighting).
        weights = gradients.mean(dim=(1, 2))  # (C,)

        weighted_activations = (weights.view(-1, 1, 1) * activations).sum(dim=0)  # (H, W)
        heatmap = F.relu(weighted_activations)  # Grad-CAM uses ReLU to keep only positive influence

        heatmap_np = heatmap.numpy().astype(np.float32)
        max_val = heatmap_np.max()
        if max_val > 1e-8:
            heatmap_np = heatmap_np / max_val
        else:
            logger.warning(
                "Grad-CAM heatmap is all-zero after ReLU (no positive "
                "gradient contribution found). Returning zero heatmap "
                "rather than fabricating normalized noise."
            )

        return heatmap_np, predicted_class_idx, predicted_class_confidence

    finally:
        hook.remove()


def heatmap_to_overlay_image(
    original_pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4
) -> Image.Image:
    """
    Resize the Grad-CAM heatmap to match the original image dimensions
    and overlay it as a color-mapped (red=high activation) semi-transparent
    layer. Returns a new PIL Image — does not mutate the input.
    """
    import cv2

    heatmap_resized = cv2.resize(
        heatmap, (original_pil_image.width, original_pil_image.height)
    )
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_arr = np.array(original_pil_image.convert("RGB")).astype(np.float32)
    overlay = (
        original_arr * (1 - alpha) + heatmap_color_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
