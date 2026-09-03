"""Deterministic model-behaviour analytics for SendCalyx.

Everything in this module is pure NumPy/stdlib so it can be unit-tested without
constructing or loading any CNN.

All quantities computed here describe *how the ensemble behaved on one input*.
They are model-behaviour diagnostics, not calibrated uncertainty, not clinical
uncertainty, and not a probability that a prediction is medically correct.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Fixed class ordering, matching the trained checkpoints.
CLASS_NAMES: List[str] = ["Kidney_stone", "Normal"]

# Guards against log(0) and division by zero.
EPS = 1e-12

# Minimum number of valid Grad-CAM maps required for cross-model aggregation.
MIN_MODELS_FOR_AGGREGATION = 2


# --------------------------------------------------------------------------- #
# Prediction consensus
# --------------------------------------------------------------------------- #
def normalized_binary_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy of a two-class distribution, normalised to ``[0, 1]``.

    ``0.0`` -> a highly one-sided distribution (e.g. 0.99 / 0.01).
    ``1.0`` -> an even 0.5 / 0.5 split.

    The raw entropy is divided by ``log(2)``, the maximum entropy of a binary
    distribution, so the result is comparable across inputs.
    """
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.size == 0:
        return 0.0

    total = float(values.sum())
    if total <= EPS:
        return 0.0

    values = np.clip(values / total, EPS, 1.0)
    entropy = float(-np.sum(values * np.log(values)))
    return float(np.clip(entropy / math.log(2.0), 0.0, 1.0))


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two discrete distributions, base 2.

    ``JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)`` with ``M = 0.5 * (P + Q)``.

    Using log base 2 bounds the result in ``[0, 1]`` for any pair of
    distributions: ``0`` when they are identical, ``1`` when they place all mass
    on different outcomes. Both inputs are renormalised and epsilon-clipped, so
    zero-probability entries are safe.
    """
    p_arr = np.asarray(list(p), dtype=np.float64)
    q_arr = np.asarray(list(q), dtype=np.float64)
    if p_arr.size != q_arr.size or p_arr.size == 0:
        return 0.0

    p_sum = float(p_arr.sum())
    q_sum = float(q_arr.sum())
    if p_sum <= EPS or q_sum <= EPS:
        return 0.0

    p_arr = np.clip(p_arr / p_sum, EPS, 1.0)
    q_arr = np.clip(q_arr / q_sum, EPS, 1.0)
    m_arr = 0.5 * (p_arr + q_arr)

    divergence = 0.5 * float(np.sum(p_arr * np.log2(p_arr / m_arr))) + 0.5 * float(
        np.sum(q_arr * np.log2(q_arr / m_arr))
    )
    return float(np.clip(divergence, 0.0, 1.0))


def compute_probability_divergence(
    individual_models: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pairwise Jensen-Shannon divergence across base-model class distributions.

    This measures how differently the base models distributed probability mass,
    which is finer-grained than the vote counts: three models can vote the same
    way while disagreeing substantially about how confident that call is.

    It is a model-behaviour diagnostic, not uncertainty. Surfaced to users as
    "probability divergence".

    Returns ``pairwise`` (one entry per model pair), ``mean``, ``max``,
    ``most_divergent_pair``, and ``available`` (False when fewer than two models
    supplied a usable distribution).
    """
    distributions: List[Tuple[str, List[float]]] = []

    for model_id, result in individual_models.items():
        probabilities = result.get("probabilities") or {}
        values = [
            float(probabilities.get(name, 0.0)) for name in CLASS_NAMES
        ]
        if not all(math.isfinite(value) for value in values):
            continue
        if sum(values) <= EPS:
            continue
        distributions.append((model_id, values))

    if len(distributions) < 2:
        return {
            "pairwise": [],
            "mean": 0.0,
            "max": 0.0,
            "most_divergent_pair": None,
            "available": False,
        }

    pairwise: List[Dict[str, Any]] = []
    for (left_id, left), (right_id, right) in itertools.combinations(distributions, 2):
        pairwise.append(
            {
                "models": [left_id, right_id],
                "value": jensen_shannon_divergence(left, right),
            }
        )

    values = [entry["value"] for entry in pairwise]
    most_divergent = max(pairwise, key=lambda entry: entry["value"])

    return {
        "pairwise": pairwise,
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
        "most_divergent_pair": list(most_divergent["models"]),
        "available": True,
    }


def compute_prediction_consensus(
    individual_models: Mapping[str, Mapping[str, Any]],
    ensemble_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Summarise how the base models voted relative to the ensemble output.

    Parameters
    ----------
    individual_models:
        Mapping of ``model_id`` to a result dict with at least ``prediction``
        (a class name) and ``confidence`` (the probability the model assigned to
        its own selected class).
    ensemble_result:
        The stacked meta-learner output, with a ``probabilities`` mapping keyed
        by class name.

    Returns
    -------
    dict with keys:
        ``votes``
            Number of base models selecting each class.
        ``num_models``
            Number of base models that produced a usable vote.
        ``majority_class`` / ``majority_votes``
            The most-voted class and its vote count.
        ``agreement_ratio``
            ``majority_votes / num_models`` (3/3 -> 1.0, 2/3 -> ~0.667).
        ``unanimous``
            True only when every base model selected the same class.
        ``disagreement_flag``
            ``not unanimous``.
        ``confidence_spread``
            Maximum minus minimum *selected-class* confidence across base
            models. Each model contributes the confidence of the class it chose,
            not the confidence of a fixed class.
        ``ensemble_margin``
            ``|P(Kidney_stone) - P(Normal)|`` from the final ensemble output.
        ``ensemble_entropy``
            Normalised binary Shannon entropy of the ensemble probabilities.
        ``ensemble_matches_majority``
            Whether the ensemble agreed with the base-model majority vote.
        ``probability_divergence``
            Pairwise Jensen-Shannon divergence across the base-model class
            distributions. See :func:`compute_probability_divergence`.
    """
    votes: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    selected_confidences: List[float] = []

    for result in individual_models.values():
        prediction = result.get("prediction")
        if prediction is None:
            continue
        votes[prediction] = votes.get(prediction, 0) + 1

        confidence = result.get("confidence")
        if confidence is not None and math.isfinite(float(confidence)):
            selected_confidences.append(float(confidence))

    num_models = sum(votes.values())

    if num_models > 0:
        majority_class = max(votes, key=lambda name: votes[name])
        majority_votes = votes[majority_class]
        agreement_ratio = majority_votes / num_models
        unanimous = majority_votes == num_models
    else:
        majority_class = None
        majority_votes = 0
        agreement_ratio = 0.0
        unanimous = False

    if selected_confidences:
        confidence_spread = max(selected_confidences) - min(selected_confidences)
    else:
        confidence_spread = 0.0

    ensemble_probabilities = ensemble_result.get("probabilities", {}) or {}
    stone_probability = float(ensemble_probabilities.get("Kidney_stone", 0.0))
    normal_probability = float(ensemble_probabilities.get("Normal", 0.0))

    ensemble_margin = abs(stone_probability - normal_probability)
    ensemble_entropy = normalized_binary_entropy(
        [stone_probability, normal_probability]
    )

    ensemble_prediction = ensemble_result.get("prediction")

    return {
        "votes": votes,
        "num_models": num_models,
        "majority_class": majority_class,
        "majority_votes": majority_votes,
        "agreement_ratio": float(agreement_ratio),
        "unanimous": bool(unanimous),
        "disagreement_flag": bool(not unanimous),
        "confidence_spread": float(confidence_spread),
        "ensemble_margin": float(ensemble_margin),
        "ensemble_entropy": float(ensemble_entropy),
        "ensemble_matches_majority": bool(
            majority_class is not None and ensemble_prediction == majority_class
        ),
        "probability_divergence": compute_probability_divergence(individual_models),
    }


# --------------------------------------------------------------------------- #
# Cross-model Grad-CAM aggregation
# --------------------------------------------------------------------------- #
def resize_map(cam: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2D map, implemented with NumPy only.

    Kept dependency-free so the aggregation logic stays unit-testable without
    OpenCV or PyTorch.
    """
    cam = np.asarray(cam, dtype=np.float32)
    target_h, target_w = int(shape[0]), int(shape[1])
    src_h, src_w = cam.shape

    if (src_h, src_w) == (target_h, target_w):
        return cam.copy()

    def axis_coords(src_size: int, target_size: int) -> np.ndarray:
        if target_size == 1 or src_size == 1:
            return np.zeros(target_size, dtype=np.float32)
        # Align corners: endpoints of the source map map to endpoints of the target.
        return np.linspace(0.0, src_size - 1.0, target_size, dtype=np.float32)

    rows = axis_coords(src_h, target_h)
    cols = axis_coords(src_w, target_w)

    row0 = np.floor(rows).astype(np.int32)
    col0 = np.floor(cols).astype(np.int32)
    row1 = np.minimum(row0 + 1, src_h - 1)
    col1 = np.minimum(col0 + 1, src_w - 1)

    row_weight = (rows - row0)[:, None]
    col_weight = (cols - col0)[None, :]

    top = cam[np.ix_(row0, col0)] * (1 - col_weight) + cam[np.ix_(row0, col1)] * col_weight
    bottom = cam[np.ix_(row1, col0)] * (1 - col_weight) + cam[np.ix_(row1, col1)] * col_weight
    return (top * (1 - row_weight) + bottom * row_weight).astype(np.float32)


def normalize_map(cam: np.ndarray) -> np.ndarray:
    """Rescale a map to ``[0, 1]``. A flat map becomes all zeros."""
    cam = np.asarray(cam, dtype=np.float32)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min <= EPS:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)


def aggregate_gradcams(
    cams: Mapping[str, Optional[np.ndarray]] | Sequence[Optional[np.ndarray]],
) -> Dict[str, Any]:
    """Combine per-model Grad-CAM maps into consensus and disagreement maps.

    Invalid entries (``None``, non-finite, wrong dimensionality) are discarded.
    Remaining maps are resized to a common shape (the largest height and width
    present), then stacked for a pixel-wise mean and variance.

    Returns
    -------
    dict with keys ``mean_cam``, ``variance_cam``, ``models_included``,
    ``model_ids``, ``shape`` and ``available``. ``available`` is False (and the
    map arrays are ``None``) when fewer than two valid maps were supplied, which
    is a normal outcome rather than a failure.
    """
    if isinstance(cams, Mapping):
        items = list(cams.items())
    else:
        items = [(str(index), cam) for index, cam in enumerate(cams)]

    valid_ids: List[str] = []
    valid_cams: List[np.ndarray] = []

    for model_id, cam in items:
        if cam is None:
            continue
        array = np.asarray(cam, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim != 2 or array.size == 0:
            continue
        if not np.all(np.isfinite(array)):
            continue
        valid_ids.append(model_id)
        valid_cams.append(array)

    if len(valid_cams) < MIN_MODELS_FOR_AGGREGATION:
        return {
            "mean_cam": None,
            "variance_cam": None,
            "models_included": len(valid_cams),
            "model_ids": valid_ids,
            "shape": None,
            "available": False,
        }

    target_shape = (
        max(cam.shape[0] for cam in valid_cams),
        max(cam.shape[1] for cam in valid_cams),
    )
    resized = [resize_map(normalize_map(cam), target_shape) for cam in valid_cams]
    stacked = np.stack(resized, axis=0)

    mean_cam = stacked.mean(axis=0).astype(np.float32)
    # Population variance across models (ddof=0): the spread of attribution at
    # each pixel between the ensemble members actually present.
    variance_cam = stacked.var(axis=0).astype(np.float32)

    return {
        "mean_cam": mean_cam,
        "variance_cam": variance_cam,
        "models_included": len(resized),
        "model_ids": valid_ids,
        "shape": tuple(int(value) for value in target_shape),
        "available": True,
    }


# --------------------------------------------------------------------------- #
# Input metadata
# --------------------------------------------------------------------------- #
def build_input_metadata(
    width: int,
    height: int,
    image_format: Optional[str],
    file_size_bytes: int,
    model_input_size: Tuple[int, int] = (299, 299),
) -> Dict[str, Any]:
    """Deterministic technical facts about the uploaded file.

    ``below_recommended_resolution`` is True when either source dimension is
    smaller than the corresponding model input dimension, meaning the image is
    upsampled before inference. It says nothing about medical image quality or
    diagnostic suitability.
    """
    width = int(width)
    height = int(height)
    target_w, target_h = int(model_input_size[0]), int(model_input_size[1])

    return {
        "width": width,
        "height": height,
        "format": (image_format or "UNKNOWN").upper(),
        "file_size_bytes": int(file_size_bytes),
        "model_input_size": [target_w, target_h],
        "below_recommended_resolution": bool(width < target_w or height < target_h),
    }
