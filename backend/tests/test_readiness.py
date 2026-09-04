"""Readiness tests for the ensemble loading path.

The stacked meta-learner consumes exactly ``3 base models x 2 classes = 6``
features, so a partial load is unusable rather than degraded. These tests pin
that behaviour: the service is ready only with all four checkpoints, and it must
never expose a randomly initialised meta-learner as though it were trained.

No real CNN is constructed and no checkpoint is read. The loading internals are
patched so failure states are exercised deterministically and in milliseconds.

Run from the repository root:

    python -m unittest discover -s backend/tests -t backend/tests
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_BASE_MODELS = ["inception_v3", "inception_resnet_v2", "xception"]


class FakeEnsemble:
    """Stand-in exposing the readiness surface the service depends on."""

    def __init__(self, base_models, meta_learner_loaded, load_errors=None):
        self.base_models = {name: object() for name in base_models}
        self.meta_learner_loaded = meta_learner_loaded
        self.load_errors = load_errors or {}
        self.eval_called = False

    @property
    def is_ready(self):
        return (
            all(name in self.base_models for name in REQUIRED_BASE_MODELS)
            and self.meta_learner_loaded
        )

    def missing_components(self):
        missing = [n for n in REQUIRED_BASE_MODELS if n not in self.base_models]
        if not self.meta_learner_loaded:
            missing.append("meta_learner")
        return missing

    def load_meta_learner(self):
        return self.meta_learner_loaded

    def eval(self):
        self.eval_called = True
        return self


class TestEnsembleReadinessRules(unittest.TestCase):
    """The is_ready contract, independent of the web layer."""

    def test_ready_only_with_all_four_checkpoints(self) -> None:
        ensemble = FakeEnsemble(REQUIRED_BASE_MODELS, meta_learner_loaded=True)
        self.assertTrue(ensemble.is_ready)
        self.assertEqual(ensemble.missing_components(), [])

    def test_missing_meta_learner_is_not_ready(self) -> None:
        ensemble = FakeEnsemble(REQUIRED_BASE_MODELS, meta_learner_loaded=False)
        self.assertFalse(ensemble.is_ready)
        self.assertEqual(ensemble.missing_components(), ["meta_learner"])

    def test_missing_one_base_model_is_not_ready(self) -> None:
        ensemble = FakeEnsemble(["inception_v3", "xception"], meta_learner_loaded=True)
        self.assertFalse(ensemble.is_ready)
        self.assertEqual(ensemble.missing_components(), ["inception_resnet_v2"])

    def test_two_base_models_are_not_a_degraded_configuration(self) -> None:
        # The meta-learner is shaped for six features; four is not "partial".
        ensemble = FakeEnsemble(["inception_v3", "xception"], meta_learner_loaded=True)
        self.assertFalse(ensemble.is_ready)

    def test_no_base_models_reports_all_missing(self) -> None:
        ensemble = FakeEnsemble([], meta_learner_loaded=False)
        self.assertFalse(ensemble.is_ready)
        self.assertEqual(
            ensemble.missing_components(), REQUIRED_BASE_MODELS + ["meta_learner"]
        )


class TestLoadModelAtomicity(unittest.TestCase):
    """load_model returns a usable model or nothing at all."""

    def _load_with(self, ensemble):
        import app_utils

        with mock.patch.object(
            app_utils, "StackedEnsembleNet", return_value=ensemble
        ):
            return app_utils.load_model()

    def test_complete_ensemble_is_returned(self) -> None:
        ensemble = FakeEnsemble(REQUIRED_BASE_MODELS, meta_learner_loaded=True)
        model, device, diagnostics = self._load_with(ensemble)

        self.assertIs(model, ensemble)
        self.assertIsNotNone(device)
        self.assertEqual(diagnostics["missing"], [])
        self.assertTrue(ensemble.eval_called)

    def test_missing_meta_learner_returns_no_model(self) -> None:
        ensemble = FakeEnsemble(
            REQUIRED_BASE_MODELS,
            meta_learner_loaded=False,
            load_errors={"meta_learner": "checkpoint not found: /app/models/meta.pth"},
        )
        model, device, diagnostics = self._load_with(ensemble)

        self.assertIsNone(model)
        self.assertIsNone(device)
        self.assertIn("meta_learner", diagnostics["missing"])
        # A randomly initialised meta-learner must never be left in eval mode
        # and handed back as if it were trained.
        self.assertFalse(ensemble.eval_called)

    def test_missing_base_model_returns_no_model(self) -> None:
        ensemble = FakeEnsemble(
            ["inception_v3", "inception_resnet_v2"], meta_learner_loaded=True
        )
        model, device, diagnostics = self._load_with(ensemble)

        self.assertIsNone(model)
        self.assertIsNone(device)
        self.assertEqual(diagnostics["missing"], ["xception"])

    def test_construction_failure_is_reported_not_raised(self) -> None:
        import app_utils

        with mock.patch.object(
            app_utils, "StackedEnsembleNet", side_effect=RuntimeError("boom")
        ):
            model, device, diagnostics = app_utils.load_model()

        self.assertIsNone(model)
        self.assertIsNone(device)
        self.assertIn("load_model", diagnostics["errors"])
        self.assertIn("RuntimeError", diagnostics["errors"]["load_model"])


class TestServiceReadinessSurface(unittest.TestCase):
    """The API reports readiness honestly and hides filesystem detail."""

    def test_is_ready_false_without_model(self) -> None:
        import app

        with mock.patch.object(app, "model", None):
            self.assertFalse(app.is_ready())

    def test_is_ready_false_for_incomplete_ensemble(self) -> None:
        import app

        incomplete = FakeEnsemble(["inception_v3"], meta_learner_loaded=True)
        with mock.patch.object(app, "model", incomplete):
            self.assertFalse(app.is_ready())

    def test_is_ready_true_for_complete_ensemble(self) -> None:
        import app

        complete = FakeEnsemble(REQUIRED_BASE_MODELS, meta_learner_loaded=True)
        with mock.patch.object(app, "model", complete):
            self.assertTrue(app.is_ready())

    def test_load_errors_do_not_leak_absolute_paths(self) -> None:
        import app

        cleaned = app.public_load_errors(
            {"meta_learner": "checkpoint not found: /srv/app/models/meta.pth"}
        )
        message = cleaned["meta_learner"]
        self.assertNotIn("/srv/app/models", message)
        self.assertIn("meta.pth", message)

    def test_public_load_errors_passes_through_plain_messages(self) -> None:
        import app

        cleaned = app.public_load_errors({"xception": "size mismatch for classifier"})
        self.assertEqual(cleaned["xception"], "size mismatch for classifier")


class TestInferenceGuard(unittest.TestCase):
    """Stacked inference refuses to run on an incomplete ensemble."""

    def test_base_probabilities_raises_when_not_ready(self) -> None:
        from architectures import StackedEnsembleNet

        ensemble = object.__new__(StackedEnsembleNet)
        ensemble.base_models = {"inception_v3": object()}
        ensemble.meta_learner_loaded = False

        with self.assertRaises(RuntimeError) as caught:
            StackedEnsembleNet.base_probabilities(ensemble, None)

        message = str(caught.exception)
        self.assertIn("not ready", message)
        self.assertIn("meta_learner", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
