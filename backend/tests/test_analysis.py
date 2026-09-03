"""Unit tests for the SendCalyx analysis helpers.

These tests deliberately construct no CNN and load no checkpoint. Everything
here runs on small synthetic inputs so the suite stays fast and deterministic.

Run from the repository root:

    python -m unittest discover -s backend/tests -t backend/tests
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import (  # noqa: E402
    aggregate_gradcams,
    build_input_metadata,
    compute_prediction_consensus,
    compute_probability_divergence,
    jensen_shannon_divergence,
    normalize_map,
    normalized_binary_entropy,
    resize_map,
)


def base_model(prediction: str, stone: float, normal: float) -> dict:
    """One synthetic base-model result in the shape the pipeline produces."""
    confidence = stone if prediction == "Kidney_stone" else normal
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {"Kidney_stone": stone, "Normal": normal},
    }


def ensemble(stone: float, normal: float) -> dict:
    return {
        "prediction": "Kidney_stone" if stone >= normal else "Normal",
        "confidence": max(stone, normal),
        "probabilities": {"Kidney_stone": stone, "Normal": normal},
    }


class TestUnanimousVote(unittest.TestCase):
    """All three base models select Kidney_stone."""

    def setUp(self) -> None:
        self.individual = {
            "inception_v3": base_model("Kidney_stone", 0.91, 0.09),
            "inception_resnet_v2": base_model("Kidney_stone", 0.84, 0.16),
            "xception": base_model("Kidney_stone", 0.77, 0.23),
        }
        self.result = compute_prediction_consensus(
            self.individual, ensemble(0.95, 0.05)
        )

    def test_votes(self) -> None:
        self.assertEqual(self.result["votes"], {"Kidney_stone": 3, "Normal": 0})
        self.assertEqual(self.result["num_models"], 3)
        self.assertEqual(self.result["majority_class"], "Kidney_stone")

    def test_agreement_ratio_is_one(self) -> None:
        self.assertAlmostEqual(self.result["agreement_ratio"], 1.0)

    def test_unanimous_and_no_disagreement(self) -> None:
        self.assertTrue(self.result["unanimous"])
        self.assertFalse(self.result["disagreement_flag"])

    def test_ensemble_matches_majority(self) -> None:
        self.assertTrue(self.result["ensemble_matches_majority"])

    def test_confidence_spread(self) -> None:
        # max selected-class confidence 0.91, min 0.77
        self.assertAlmostEqual(self.result["confidence_spread"], 0.14, places=7)

    def test_ensemble_margin(self) -> None:
        self.assertAlmostEqual(self.result["ensemble_margin"], 0.90, places=7)


class TestSplitVote(unittest.TestCase):
    """Two models select Kidney_stone, one selects Normal."""

    def setUp(self) -> None:
        self.individual = {
            "inception_v3": base_model("Kidney_stone", 0.91, 0.09),
            "inception_resnet_v2": base_model("Kidney_stone", 0.84, 0.16),
            "xception": base_model("Normal", 0.43, 0.57),
        }
        self.result = compute_prediction_consensus(
            self.individual, ensemble(0.71, 0.29)
        )

    def test_votes(self) -> None:
        self.assertEqual(self.result["votes"], {"Kidney_stone": 2, "Normal": 1})
        self.assertEqual(self.result["majority_votes"], 2)

    def test_agreement_ratio_is_two_thirds(self) -> None:
        self.assertAlmostEqual(self.result["agreement_ratio"], 2 / 3, places=7)

    def test_disagreement_flag_set(self) -> None:
        self.assertFalse(self.result["unanimous"])
        self.assertTrue(self.result["disagreement_flag"])

    def test_confidence_spread_uses_selected_class(self) -> None:
        # Selected-class confidences are 0.91, 0.84 and 0.57 -> spread 0.34.
        self.assertAlmostEqual(self.result["confidence_spread"], 0.34, places=7)

    def test_ensemble_margin(self) -> None:
        self.assertAlmostEqual(self.result["ensemble_margin"], 0.42, places=7)

    def test_entropy_is_finite_and_bounded(self) -> None:
        entropy = self.result["ensemble_entropy"]
        self.assertTrue(math.isfinite(entropy))
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)


class TestPredictiveEntropy(unittest.TestCase):
    def test_low_for_one_sided_distribution(self) -> None:
        self.assertLess(normalized_binary_entropy([0.99, 0.01]), 0.1)
        self.assertLess(normalized_binary_entropy([0.001, 0.999]), 0.02)

    def test_high_near_even_split(self) -> None:
        self.assertGreater(normalized_binary_entropy([0.5, 0.5]), 0.99)
        self.assertGreater(normalized_binary_entropy([0.52, 0.48]), 0.99)

    def test_maximum_is_exactly_one(self) -> None:
        self.assertAlmostEqual(normalized_binary_entropy([0.5, 0.5]), 1.0, places=9)

    def test_monotonic_with_certainty(self) -> None:
        values = [
            normalized_binary_entropy([p, 1 - p]) for p in (0.5, 0.7, 0.9, 0.99)
        ]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_degenerate_inputs_are_safe(self) -> None:
        self.assertAlmostEqual(normalized_binary_entropy([1.0, 0.0]), 0.0, places=9)
        self.assertEqual(normalized_binary_entropy([0.0, 0.0]), 0.0)
        self.assertEqual(normalized_binary_entropy([]), 0.0)

    def test_consensus_entropy_matches_helper(self) -> None:
        result = compute_prediction_consensus(
            {"m": base_model("Normal", 0.4, 0.6)}, ensemble(0.5, 0.5)
        )
        self.assertAlmostEqual(result["ensemble_entropy"], 1.0, places=9)


class TestJensenShannonDivergence(unittest.TestCase):
    def test_identical_distributions_are_zero(self) -> None:
        self.assertAlmostEqual(jensen_shannon_divergence([0.7, 0.3], [0.7, 0.3]), 0.0, places=9)
        self.assertAlmostEqual(jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]), 0.0, places=9)

    def test_symmetry(self) -> None:
        p, q = [0.9, 0.1], [0.25, 0.75]
        self.assertAlmostEqual(
            jensen_shannon_divergence(p, q), jensen_shannon_divergence(q, p), places=12
        )

    def test_opposed_one_hot_distributions_approach_one(self) -> None:
        self.assertGreater(jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]), 0.999)

    def test_bounded_in_unit_interval(self) -> None:
        for p in (0.0, 0.01, 0.3, 0.5, 0.87, 1.0):
            for q in (0.0, 0.05, 0.5, 0.99, 1.0):
                value = jensen_shannon_divergence([p, 1 - p], [q, 1 - q])
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_known_value_for_uniform_versus_one_hot(self) -> None:
        # M = [0.75, 0.25]; JSD = 0.5*log2(4/3) + 0.5*(0.5*log2(2/3) + 0.5*log2(2))
        expected = 0.5 * math.log2(1 / 0.75) + 0.5 * (
            0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)
        )
        self.assertAlmostEqual(
            jensen_shannon_divergence([1.0, 0.0], [0.5, 0.5]), expected, places=9
        )
        self.assertAlmostEqual(expected, 0.3112781244, places=9)

    def test_grows_with_separation(self) -> None:
        values = [
            jensen_shannon_divergence([0.5, 0.5], [q, 1 - q]) for q in (0.5, 0.6, 0.8, 0.99)
        ]
        self.assertEqual(values, sorted(values))

    def test_zero_probabilities_stay_finite(self) -> None:
        value = jensen_shannon_divergence([0.0, 1.0], [1e-12, 1 - 1e-12])
        self.assertTrue(math.isfinite(value))

    def test_degenerate_inputs_are_safe(self) -> None:
        self.assertEqual(jensen_shannon_divergence([], []), 0.0)
        self.assertEqual(jensen_shannon_divergence([0.0, 0.0], [0.5, 0.5]), 0.0)
        self.assertEqual(jensen_shannon_divergence([0.5, 0.5], [0.2, 0.3, 0.5]), 0.0)


class TestProbabilityDivergence(unittest.TestCase):
    def test_identical_models_give_zero(self) -> None:
        models = {
            "a": base_model("Kidney_stone", 0.8, 0.2),
            "b": base_model("Kidney_stone", 0.8, 0.2),
            "c": base_model("Kidney_stone", 0.8, 0.2),
        }
        result = compute_probability_divergence(models)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["pairwise"]), 3)
        self.assertAlmostEqual(result["mean"], 0.0, places=9)
        self.assertAlmostEqual(result["max"], 0.0, places=9)

    def test_pair_count_and_mean_max(self) -> None:
        models = {
            "a": base_model("Kidney_stone", 1.0, 0.0),
            "b": base_model("Kidney_stone", 1.0, 0.0),
            "c": base_model("Normal", 0.0, 1.0),
        }
        result = compute_probability_divergence(models)
        values = sorted(entry["value"] for entry in result["pairwise"])
        self.assertEqual(len(values), 3)
        # a-b identical (0), a-c and b-c fully opposed (1 each).
        self.assertAlmostEqual(values[0], 0.0, places=6)
        self.assertAlmostEqual(values[1], 1.0, places=3)
        self.assertAlmostEqual(values[2], 1.0, places=3)
        self.assertAlmostEqual(result["mean"], sum(values) / 3, places=9)
        self.assertAlmostEqual(result["max"], max(values), places=9)

    def test_most_divergent_pair_is_identified(self) -> None:
        models = {
            "inception_v3": base_model("Kidney_stone", 0.95, 0.05),
            "inception_resnet_v2": base_model("Kidney_stone", 0.92, 0.08),
            "xception": base_model("Normal", 0.10, 0.90),
        }
        result = compute_probability_divergence(models)
        self.assertIn("xception", result["most_divergent_pair"])
        self.assertEqual(len(result["most_divergent_pair"]), 2)

    def test_single_model_is_unavailable(self) -> None:
        result = compute_probability_divergence({"a": base_model("Normal", 0.2, 0.8)})
        self.assertFalse(result["available"])
        self.assertEqual(result["pairwise"], [])
        self.assertEqual(result["mean"], 0.0)
        self.assertIsNone(result["most_divergent_pair"])

    def test_empty_input_is_unavailable(self) -> None:
        self.assertFalse(compute_probability_divergence({})["available"])

    def test_models_without_usable_probabilities_are_skipped(self) -> None:
        models = {
            "a": base_model("Kidney_stone", 0.8, 0.2),
            "b": base_model("Kidney_stone", 0.7, 0.3),
            "c": {"prediction": "Normal", "confidence": 0.5, "probabilities": {}},
        }
        result = compute_probability_divergence(models)
        self.assertEqual(len(result["pairwise"]), 1)
        self.assertEqual(result["pairwise"][0]["models"], ["a", "b"])

    def test_exposed_through_consensus(self) -> None:
        models = {
            "a": base_model("Kidney_stone", 0.9, 0.1),
            "b": base_model("Normal", 0.2, 0.8),
        }
        consensus = compute_prediction_consensus(models, ensemble(0.6, 0.4))
        divergence = consensus["probability_divergence"]
        self.assertTrue(divergence["available"])
        self.assertTrue(math.isfinite(divergence["mean"]))
        self.assertGreater(divergence["mean"], 0.0)


class TestGradCamAggregation(unittest.TestCase):
    def setUp(self) -> None:
        # Three 2x2 maps, already in [0, 1] and not flat, so normalisation is
        # the identity and the expected mean/variance can be computed by hand.
        self.cams = {
            "a": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "b": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            "c": np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        }
        self.result = aggregate_gradcams(self.cams)

    def test_available_and_models_included(self) -> None:
        self.assertTrue(self.result["available"])
        self.assertEqual(self.result["models_included"], 3)
        self.assertEqual(self.result["model_ids"], ["a", "b", "c"])

    def test_output_shapes(self) -> None:
        self.assertEqual(self.result["shape"], (2, 2))
        self.assertEqual(self.result["mean_cam"].shape, (2, 2))
        self.assertEqual(self.result["variance_cam"].shape, (2, 2))

    def test_pixelwise_mean(self) -> None:
        expected = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(self.result["mean_cam"], expected, atol=1e-6)

    def test_pixelwise_variance(self) -> None:
        # Column 0 lower pixel: values 0.0, 1.0, 0.5 -> mean 0.5, population
        # variance = (0.25 + 0.25 + 0.0) / 3 = 1/6.
        expected = np.array([[0.0, 0.0], [1 / 6, 1 / 6]], dtype=np.float32)
        np.testing.assert_allclose(self.result["variance_cam"], expected, atol=1e-6)

    def test_identical_maps_have_zero_variance(self) -> None:
        cam = np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        result = aggregate_gradcams({"a": cam, "b": cam.copy()})
        np.testing.assert_allclose(
            result["variance_cam"], np.zeros((2, 2), dtype=np.float32), atol=1e-6
        )
        np.testing.assert_allclose(result["mean_cam"], cam, atol=1e-6)


class TestGradCamAggregationEdgeCases(unittest.TestCase):
    def test_none_maps_are_discarded(self) -> None:
        cam = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        result = aggregate_gradcams({"a": cam, "b": None, "c": cam.copy()})
        self.assertTrue(result["available"])
        self.assertEqual(result["models_included"], 2)
        self.assertEqual(result["model_ids"], ["a", "c"])

    def test_single_valid_map_is_unavailable(self) -> None:
        result = aggregate_gradcams({"a": np.zeros((4, 4), np.float32), "b": None})
        self.assertFalse(result["available"])
        self.assertEqual(result["models_included"], 1)
        self.assertIsNone(result["mean_cam"])
        self.assertIsNone(result["variance_cam"])

    def test_no_valid_maps_is_unavailable(self) -> None:
        result = aggregate_gradcams({"a": None, "b": None, "c": None})
        self.assertFalse(result["available"])
        self.assertEqual(result["models_included"], 0)

    def test_non_finite_maps_are_discarded(self) -> None:
        good = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        bad = np.array([[np.nan, 1.0], [1.0, 0.0]], dtype=np.float32)
        result = aggregate_gradcams({"a": good, "b": bad, "c": good.copy()})
        self.assertEqual(result["models_included"], 2)
        self.assertTrue(np.all(np.isfinite(result["mean_cam"])))

    def test_mismatched_shapes_are_resized_to_the_largest(self) -> None:
        # 8x8 (InceptionV3 / InceptionResNetV2) alongside 10x10 (Xception).
        small = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        large = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        result = aggregate_gradcams({"small": small, "large": large})
        self.assertTrue(result["available"])
        self.assertEqual(result["shape"], (10, 10))
        self.assertEqual(result["mean_cam"].shape, (10, 10))
        self.assertEqual(result["variance_cam"].shape, (10, 10))

    def test_sequence_input_is_accepted(self) -> None:
        cam = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        result = aggregate_gradcams([cam, cam.copy(), None])
        self.assertEqual(result["models_included"], 2)
        self.assertEqual(result["model_ids"], ["0", "1"])


class TestMapHelpers(unittest.TestCase):
    def test_resize_preserves_corners(self) -> None:
        cam = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        resized = resize_map(cam, (4, 4))
        self.assertEqual(resized.shape, (4, 4))
        self.assertAlmostEqual(float(resized[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(resized[0, -1]), 1.0, places=5)
        self.assertAlmostEqual(float(resized[-1, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(resized[-1, -1]), 3.0, places=5)

    def test_resize_is_identity_for_same_shape(self) -> None:
        cam = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(resize_map(cam, (2, 2)), cam)

    def test_normalize_map_rescales_to_unit_range(self) -> None:
        cam = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
        normalized = normalize_map(cam)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_normalize_flat_map_is_zeros(self) -> None:
        cam = np.full((3, 3), 0.7, dtype=np.float32)
        np.testing.assert_allclose(normalize_map(cam), np.zeros((3, 3), np.float32))


class TestInputMetadata(unittest.TestCase):
    def test_standard_resolution(self) -> None:
        metadata = build_input_metadata(512, 512, "png", 428371)
        self.assertEqual(metadata["width"], 512)
        self.assertEqual(metadata["height"], 512)
        self.assertEqual(metadata["format"], "PNG")
        self.assertEqual(metadata["file_size_bytes"], 428371)
        self.assertEqual(metadata["model_input_size"], [299, 299])
        self.assertFalse(metadata["below_recommended_resolution"])

    def test_flag_set_when_either_dimension_is_small(self) -> None:
        self.assertTrue(
            build_input_metadata(200, 512, "JPEG", 1000)["below_recommended_resolution"]
        )
        self.assertTrue(
            build_input_metadata(512, 128, "JPEG", 1000)["below_recommended_resolution"]
        )

    def test_exact_model_input_size_is_not_flagged(self) -> None:
        metadata = build_input_metadata(299, 299, "PNG", 1000)
        self.assertFalse(metadata["below_recommended_resolution"])

    def test_missing_format_is_reported_as_unknown(self) -> None:
        self.assertEqual(build_input_metadata(400, 400, None, 10)["format"], "UNKNOWN")


class TestConsensusEdgeCases(unittest.TestCase):
    def test_empty_input_does_not_raise(self) -> None:
        result = compute_prediction_consensus({}, ensemble(0.5, 0.5))
        self.assertEqual(result["num_models"], 0)
        self.assertEqual(result["agreement_ratio"], 0.0)
        self.assertFalse(result["unanimous"])
        self.assertIsNone(result["majority_class"])

    def test_single_model_is_trivially_unanimous(self) -> None:
        result = compute_prediction_consensus(
            {"inception_v3": base_model("Normal", 0.2, 0.8)}, ensemble(0.2, 0.8)
        )
        self.assertTrue(result["unanimous"])
        self.assertAlmostEqual(result["agreement_ratio"], 1.0)
        self.assertAlmostEqual(result["confidence_spread"], 0.0)

    def test_all_values_are_json_serialisable_floats(self) -> None:
        result = compute_prediction_consensus(
            {"a": base_model("Normal", 0.2, 0.8)}, ensemble(0.2, 0.8)
        )
        for key in (
            "agreement_ratio",
            "confidence_spread",
            "ensemble_margin",
            "ensemble_entropy",
        ):
            self.assertIsInstance(result[key], float)
            self.assertTrue(math.isfinite(result[key]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
