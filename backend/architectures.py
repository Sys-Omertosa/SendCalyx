"""CNN backbones, custom classifier head, and the stacked ensemble.

The architecture definitions here are checkpoint-compatible with the trained
weights shipped in ``backend/models/``. Tensor shapes, layer ordering, class
ordering and the meta-learner topology must not be changed without retraining,
because the saved ``state_dict`` files are loaded with ``strict=True``.

Class ordering (fixed by ``torchvision.datasets.ImageFolder`` alphabetical
sorting at training time):

    index 0 -> "Kidney_stone"
    index 1 -> "Normal"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import timm
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = ["Kidney_stone", "Normal"]
MODEL_INPUT_SIZE = (299, 299)

# Checkpoint filenames, keyed by the model id used throughout the API.
CHECKPOINT_FILES: Dict[str, str] = {
    "inception_v3": "inception_v3_kidney_stone_model.pth",
    "inception_resnet_v2": "inceptionresnetv2_kidney_stone_model.pth",
    "xception": "xception_kidney_stone_model.pth",
}
META_LEARNER_FILE = "stacked_ensemble_meta_learner.pth"

# Human-readable labels for the frontend.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "inception_v3": "InceptionV3",
    "inception_resnet_v2": "InceptionResNetV2",
    "xception": "Xception",
}


def get_model_dir() -> Path:
    """Resolve the checkpoint directory.

    Defaults to ``<this file's directory>/models`` so the service can be started
    from any working directory. Overridable with ``SENDCALYX_MODEL_DIR``.
    """
    override = os.environ.get("SENDCALYX_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "models"


class CustomClassifier(nn.Module):
    """Classifier head attached to each frozen feature-extraction backbone.

    ``input_features -> 256 -> BN -> ReLU -> Dropout(0.2) -> 128 -> BN -> ReLU -> 2``
    """

    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FeatureExtractionModel(nn.Module):
    """A backbone that emits a feature vector, plus the classifier head."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        # InceptionV3 returns (logits, aux_logits) while training.
        if isinstance(features, tuple):
            features = features[0]

        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        return self.classifier(features)


def build_inception_v3() -> nn.Module:
    """InceptionV3 backbone configured exactly as during training.

    Training used ``weights=Inception_V3_Weights.DEFAULT``, which torchvision
    expands to ``transform_input=True, aux_logits=True, init_weights=False``.
    Those flags are passed explicitly here so the module is identical without
    downloading ImageNet weights that the kidney-stone checkpoint overwrites in
    full anyway.
    """
    backbone = models.inception_v3(
        weights=None,
        transform_input=True,
        aux_logits=True,
        init_weights=False,
    )
    backbone.fc = nn.Identity()
    backbone.AuxLogits.fc = nn.Identity()
    return backbone


def build_backbone(model_id: str) -> nn.Module:
    """Construct one base-model backbone with its trained output dimensionality."""
    if model_id == "inception_v3":
        return build_inception_v3()
    if model_id == "inception_resnet_v2":
        return timm.create_model("inception_resnet_v2", pretrained=False, num_classes=0)
    if model_id == "xception":
        # The original Xception. Recent timm releases renamed it to
        # "legacy_xception" and keep "xception" as a deprecated alias.
        #
        # Note: the training run created this backbone without
        # ``pretrained=True``, i.e. from random initialisation rather than
        # ImageNet weights. Preserved deliberately - the shipped checkpoint was
        # produced under that setting.
        try:
            return timm.create_model("legacy_xception", pretrained=False, num_classes=0)
        except RuntimeError:
            return timm.create_model("xception", pretrained=False, num_classes=0)
    raise ValueError(f"Unknown model id: {model_id}")


def backbone_feature_dim(model_id: str, backbone: nn.Module) -> int:
    if model_id == "inception_v3":
        return 2048
    return int(backbone.num_features)


class StackedEnsembleNet(nn.Module):
    """Three CNNs feeding class probabilities into a small stacked meta-learner.

    The meta-learner consumes ``3 models x 2 class probabilities = 6`` features:
    ``6 -> 512 -> ReLU -> Dropout(0.2) -> 128 -> ReLU -> 2``.
    """

    def __init__(self, device: torch.device, model_dir: str | Path | None = None) -> None:
        super().__init__()
        self.device = device
        self.model_dir = Path(model_dir) if model_dir is not None else get_model_dir()

        # Plain dict (not ModuleDict) so ``self.parameters()`` stays scoped to
        # the meta-learner, matching how the ensemble was trained.
        self.base_models: Dict[str, nn.Module] = {}
        self.load_errors: Dict[str, str] = {}
        self.meta_learner_loaded = False
        self._load_base_models()

        self.meta_learner = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        ).to(device)

        logger.info("StackedEnsembleNet ready with %d base models", len(self.base_models))

    def _load_base_models(self) -> None:
        for model_id, checkpoint_name in CHECKPOINT_FILES.items():
            checkpoint_path = self.model_dir / checkpoint_name
            try:
                backbone = build_backbone(model_id)
                for param in backbone.parameters():
                    param.requires_grad = False

                classifier = CustomClassifier(
                    input_features=backbone_feature_dim(model_id, backbone)
                )
                model = FeatureExtractionModel(backbone, classifier).to(self.device)

                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

                state_dict = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True
                )
                model.load_state_dict(state_dict)
                model.eval()

                self.base_models[model_id] = model
                logger.info("Loaded %s from %s", model_id, checkpoint_path.name)
            except Exception as exc:  # noqa: BLE001 - one bad model must not kill the service
                self.load_errors[model_id] = str(exc)
                logger.error("Failed to load %s: %s", model_id, exc)

    @property
    def is_ready(self) -> bool:
        """True only when the whole ensemble is usable for inference.

        The meta-learner consumes exactly ``3 models x 2 classes = 6`` features,
        so a partial set of base models is not a degraded configuration, it is an
        unusable one. A missing meta-learner checkpoint is equally fatal: the
        randomly initialised layer would still produce confident-looking output.
        """
        return (
            all(model_id in self.base_models for model_id in CHECKPOINT_FILES)
            and self.meta_learner_loaded
        )

    def missing_components(self) -> List[str]:
        """Required checkpoints that did not load, for health reporting."""
        missing = [
            model_id for model_id in CHECKPOINT_FILES if model_id not in self.base_models
        ]
        if not self.meta_learner_loaded:
            missing.append("meta_learner")
        return missing

    def load_meta_learner(self) -> bool:
        """Load the trained meta-learner weights. Returns True on success."""
        checkpoint_path = self.model_dir / META_LEARNER_FILE
        if not checkpoint_path.exists():
            self.load_errors["meta_learner"] = f"checkpoint not found: {checkpoint_path}"
            logger.error("Meta-learner checkpoint missing: %s", checkpoint_path)
            return False
        try:
            state_dict = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.meta_learner.load_state_dict(state_dict)
            self.meta_learner_loaded = True
            logger.info("Loaded meta-learner from %s", checkpoint_path.name)
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_errors["meta_learner"] = str(exc)
            logger.error("Failed to load meta-learner: %s", exc)
            return False

    def base_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenated softmax outputs of every base model.

        Refuses to run unless the full ensemble is present: the meta-learner is
        shaped for exactly six features, and a randomly initialised one would
        emit plausible-looking probabilities from untrained weights.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Ensemble is not ready; missing: "
                + ", ".join(self.missing_components())
            )

        outputs = []
        for model in self.base_models.values():
            model.eval()
            outputs.append(torch.softmax(model(x), dim=1))
        if not outputs:
            raise RuntimeError("No base models available for ensemble inference")
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.base_probabilities(x)
        return self.meta_learner(features)

    def parameters(self, recurse: bool = True):
        """Only the meta-learner is trainable; base models stay frozen."""
        return self.meta_learner.parameters(recurse=recurse)

    def eval(self):  # type: ignore[override]
        super().eval()
        for model in self.base_models.values():
            model.eval()
        return self
