"""Inference pipeline: preprocessing, Grad-CAM, and response assembly."""

from __future__ import annotations

import base64
import gc
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

matplotlib.use("Agg")  # headless rendering for the colormaps below

from analysis import (
    aggregate_gradcams,
    build_input_metadata,
    compute_prediction_consensus,
)
from architectures import (
    CLASS_NAMES,
    MODEL_DISPLAY_NAMES,
    MODEL_INPUT_SIZE,
    StackedEnsembleNet,
)

logger = logging.getLogger(__name__)

# Grad-CAM target layer per backbone: the last spatial feature map before
# global pooling for each architecture.
TARGET_LAYERS: Dict[str, str] = {
    "inception_v3": "backbone.Mixed_7c",
    "inception_resnet_v2": "backbone.conv2d_7b",
    "xception": "backbone.conv4",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class PredictionResponse(BaseModel):
    """Shape of a successful ``POST /predict`` response."""

    ensemble: Dict[str, Any] = Field(default_factory=dict)
    individual_models: Dict[str, Any] = Field(default_factory=dict)
    consensus: Dict[str, Any] = Field(default_factory=dict)
    xai_consensus: Dict[str, Any] = Field(default_factory=dict)
    input_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = 0.0
    num_models: int = 0
    success: bool = False
    message: str = ""


class GradCAM:
    """Grad-CAM for a single model, scoped to one target layer.

    Hooks are registered on construction and must be released with
    :meth:`remove_hooks`; the class also works as a context manager.
    """

    def __init__(self, model: nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []
        self._register_hooks()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove_hooks()

    def _find_target_layer(self) -> Optional[nn.Module]:
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module

        # Fallback: the last convolution in the network.
        last_conv = None
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            logger.warning(
                "Grad-CAM target layer %s not found; using last Conv2d",
                self.target_layer_name,
            )
        return last_conv

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        target_layer = self._find_target_layer()
        if target_layer is None:
            logger.error("No Grad-CAM target layer available for this model")
            return
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Return a normalised ``[0, 1]`` attribution map, or None on failure."""
        if not self.hooks:
            return None
        try:
            self.model.zero_grad(set_to_none=True)
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())

            output[0, class_idx].backward()

            if self.gradients is None or self.activations is None:
                return None

            gradients = self.gradients[0]
            activations = self.activations[0]

            # Channel weights = global average pooled gradients.
            weights = gradients.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))

            cam_max = float(cam.max())
            if cam_max > 0:
                cam = cam / cam_max

            return cam.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - a failed CAM must not fail the prediction
            logger.warning("Grad-CAM generation failed: %s", exc)
            return None
        finally:
            self.model.zero_grad(set_to_none=True)
            self.gradients = None
            self.activations = None

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients = None
        self.activations = None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize to 299x299, convert to a tensor, and apply ImageNet normalisation."""
    rgb_image = image.convert("RGB")
    tensor = _PREPROCESS(rgb_image).unsqueeze(0)
    if rgb_image is not image:
        rgb_image.close()
    return tensor


def image_to_base64(image_array: np.ndarray) -> Optional[str]:
    """Encode an RGB uint8 array as a base64 PNG string."""
    try:
        array = image_array
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255).astype(np.uint8)

        with io.BytesIO() as buffer:
            pil_image = Image.fromarray(array)
            pil_image.save(buffer, format="PNG", optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            pil_image.close()
        return encoded
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to encode image to base64: %s", exc)
        return None


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image.resize(MODEL_INPUT_SIZE))
    if rgb_image is not image:
        rgb_image.close()
    return array


def create_heatmap_overlay(
    original_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Optional[np.ndarray]:
    """Blend a colormapped attribution map over the resized source image."""
    try:
        cam_2d = np.squeeze(np.asarray(cam, dtype=np.float32))
        if cam_2d.ndim != 2:
            return None

        cam_resized = cv2.resize(cam_2d, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        img_array = _image_to_rgb_array(original_image)

        heatmap = matplotlib.colormaps[colormap](cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        if heatmap.shape[:2] != img_array.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

        return cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build heatmap overlay: %s", exc)
        return None


def load_model(
    model_dir: Optional[str] = None,
) -> Tuple[Optional[StackedEnsembleNet], Optional[torch.device]]:
    """Build the ensemble, load every checkpoint, and return it in eval mode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = StackedEnsembleNet(device, model_dir=model_dir)
        model.load_meta_learner()
        model.eval()
        logger.info(
            "Model ready on %s with base models: %s",
            device,
            ", ".join(model.base_models) or "none",
        )
        return model, device
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model failed to load: %s", exc)
        return None, None


def _run_base_models(
    ensemble_model: StackedEnsembleNet,
    input_tensor: torch.Tensor,
    image: Image.Image,
) -> Tuple[Dict[str, Any], Dict[str, Optional[np.ndarray]]]:
    """Per-model probabilities plus a Grad-CAM overlay for each base model."""
    individual_results: Dict[str, Any] = {}
    raw_cams: Dict[str, Optional[np.ndarray]] = {}

    for model_id, base_model in ensemble_model.base_models.items():
        base_model.eval()

        with torch.no_grad():
            outputs = base_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = int(torch.argmax(probabilities).item())
            probability_values = [float(value) for value in probabilities]
            del outputs, probabilities

        cam = None
        with GradCAM(base_model, TARGET_LAYERS.get(model_id, "")) as gradcam:
            cam_input = input_tensor.clone().detach().requires_grad_(True)
            cam = gradcam.generate_cam(cam_input, predicted_class)
            del cam_input

        raw_cams[model_id] = cam

        overlay_base64 = None
        if cam is not None:
            overlay = create_heatmap_overlay(image, cam)
            if overlay is not None:
                overlay_base64 = image_to_base64(overlay)

        individual_results[model_id] = {
            "display_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": probability_values[predicted_class],
            "probabilities": {
                CLASS_NAMES[0]: probability_values[0],
                CLASS_NAMES[1]: probability_values[1],
            },
            "gradcam_overlay": overlay_base64,
            "gradcam_available": overlay_base64 is not None,
        }

    return individual_results, raw_cams


def _build_xai_consensus(
    raw_cams: Dict[str, Optional[np.ndarray]], image: Image.Image
) -> Dict[str, Any]:
    """Render mean and variance attribution maps across the available models."""
    aggregation = aggregate_gradcams(raw_cams)

    if not aggregation["available"]:
        return {
            "mean_gradcam": None,
            "variance_gradcam": None,
            "models_included": aggregation["models_included"],
            "model_ids": aggregation["model_ids"],
            "available": False,
            "message": "Cross-model attribution needs at least two valid Grad-CAM maps.",
        }

    mean_cam = aggregation["mean_cam"]
    variance_cam = aggregation["variance_cam"]

    # Variance is small in absolute terms; rescale to [0, 1] for display and
    # report the true maximum so the visualisation stays interpretable.
    variance_max = float(variance_cam.max())
    variance_display = variance_cam / variance_max if variance_max > 0 else variance_cam

    mean_overlay = create_heatmap_overlay(image, mean_cam, alpha=0.45, colormap="jet")
    variance_overlay = create_heatmap_overlay(
        image, variance_display, alpha=0.45, colormap="inferno"
    )

    return {
        "mean_gradcam": image_to_base64(mean_overlay) if mean_overlay is not None else None,
        "variance_gradcam": (
            image_to_base64(variance_overlay) if variance_overlay is not None else None
        ),
        "models_included": aggregation["models_included"],
        "model_ids": aggregation["model_ids"],
        "max_variance": variance_max,
        "mean_attribution": float(mean_cam.mean()),
        "available": True,
        "message": "",
    }


def predict_all_models(
    ensemble_model: StackedEnsembleNet,
    device: torch.device,
    image: Image.Image,
    file_size_bytes: int = 0,
) -> Dict[str, Any]:
    """Run the full SendCalyx analysis for one image.

    Returns the complete ``/predict`` payload. Errors are reported through the
    ``success`` / ``message`` fields rather than as exceptions, so a partially
    degraded run still yields a usable response.
    """
    start_time = time.time()
    input_tensor = None

    try:
        source_width, source_height = image.size
        source_format = image.format

        input_tensor = preprocess_image(image).to(device)

        individual_results, raw_cams = _run_base_models(
            ensemble_model, input_tensor, image
        )

        with torch.no_grad():
            ensemble_outputs = ensemble_model(input_tensor)
            ensemble_probabilities = torch.softmax(ensemble_outputs, dim=1)[0]
            ensemble_class = int(torch.argmax(ensemble_probabilities).item())
            ensemble_values = [float(value) for value in ensemble_probabilities]
            del ensemble_outputs, ensemble_probabilities

        ensemble_result = {
            "prediction": CLASS_NAMES[ensemble_class],
            "confidence": ensemble_values[ensemble_class],
            "probabilities": {
                CLASS_NAMES[0]: ensemble_values[0],
                CLASS_NAMES[1]: ensemble_values[1],
            },
        }

        consensus = compute_prediction_consensus(individual_results, ensemble_result)
        xai_consensus = _build_xai_consensus(raw_cams, image)
        input_metadata = build_input_metadata(
            width=source_width,
            height=source_height,
            image_format=source_format,
            file_size_bytes=file_size_bytes,
            model_input_size=MODEL_INPUT_SIZE,
        )

        return {
            "ensemble": ensemble_result,
            "individual_models": individual_results,
            "consensus": consensus,
            "xai_consensus": xai_consensus,
            "input_metadata": input_metadata,
            "processing_time": time.time() - start_time,
            "num_models": len(ensemble_model.base_models) + 1,
            "success": True,
            "message": "Prediction completed successfully",
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return {
            "ensemble": {},
            "individual_models": {},
            "consensus": {},
            "xai_consensus": {},
            "input_metadata": {},
            "processing_time": time.time() - start_time,
            "num_models": 0,
            "success": False,
            "message": f"Prediction failed: {type(exc).__name__}",
        }
    finally:
        if input_tensor is not None:
            del input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
