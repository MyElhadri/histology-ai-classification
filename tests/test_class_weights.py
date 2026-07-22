"""Tests for the class weight computation module.

All tests are **independent of TensorFlow** and can run locally.
"""

from pathlib import Path

import pytest

from src.training.class_weights import compute_balanced_class_weights

# ---------------------------------------------------------------------------
# Path to the real folds manifest
# ---------------------------------------------------------------------------
FOLDS_PATH = Path("data/manifests/densenet121_folds.csv")


# ===================================================================
# Unit tests with synthetic data
# ===================================================================


class TestSyntheticData:
    """Tests using hand-crafted label distributions."""

    def test_small_class_gets_higher_weight(self) -> None:
        """A class with fewer samples must receive a higher weight."""
        # class 0: 2 samples, class 1: 8 samples → total=10, num_classes=2
        labels = [0, 0] + [1] * 8
        weights = compute_balanced_class_weights(labels, num_classes=2)
        assert weights[0] > weights[1]

    def test_equal_classes_get_equal_weight(self) -> None:
        """Two classes with the same count must get identical weights."""
        labels = [0] * 5 + [1] * 5
        weights = compute_balanced_class_weights(labels, num_classes=2)
        assert weights[0] == weights[1]

    def test_perfectly_balanced_weight_is_one(self) -> None:
        """When all classes have the same count, every weight should be 1.0."""
        labels = [0, 0, 1, 1, 2, 2]
        weights = compute_balanced_class_weights(labels, num_classes=3)
        for w in weights.values():
            assert w == pytest.approx(1.0)

    def test_all_keys_are_int(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]
        weights = compute_balanced_class_weights(labels, num_classes=3)
        for key in weights.keys():
            assert isinstance(key, int), f"Key {key!r} is {type(key).__name__}, expected int"

    def test_all_values_are_positive_float(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]
        weights = compute_balanced_class_weights(labels, num_classes=3)
        for key, val in weights.items():
            assert isinstance(val, float), f"Value for class {key} is {type(val).__name__}"
            assert val > 0, f"Weight for class {key} is not positive: {val}"

    def test_deterministic(self) -> None:
        """Two calls with the same input must produce identical output."""
        labels = [0] * 3 + [1] * 7 + [2] * 10
        w1 = compute_balanced_class_weights(labels, num_classes=3)
        w2 = compute_balanced_class_weights(labels, num_classes=3)
        assert w1 == w2

    def test_missing_class_raises_value_error(self) -> None:
        """If an expected class is absent, a ValueError must be raised."""
        labels = [0, 0, 1, 1]  # missing class 2
        with pytest.raises(ValueError, match="missing"):
            compute_balanced_class_weights(labels, num_classes=3)

    def test_input_not_mutated(self) -> None:
        """The function must not modify the input list."""
        labels = [0, 1, 2, 0, 1, 2]
        original = labels.copy()
        compute_balanced_class_weights(labels, num_classes=3)
        assert labels == original

    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_balanced_class_weights([], num_classes=3)

    def test_invalid_num_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            compute_balanced_class_weights([0, 1], num_classes=0)

    def test_formula_correctness(self) -> None:
        """Verify the exact formula: weight_c = n_total / (n_classes * n_c)."""
        # class 0: 2, class 1: 3, class 2: 5 → total = 10, n_classes = 3
        labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
        weights = compute_balanced_class_weights(labels, num_classes=3)
        assert weights[0] == pytest.approx(10 / (3 * 2))   # 1.6667
        assert weights[1] == pytest.approx(10 / (3 * 3))   # 1.1111
        assert weights[2] == pytest.approx(10 / (3 * 5))   # 0.6667


# ===================================================================
# Integration tests with the real folds manifest
# ===================================================================


@pytest.fixture
def folds_df():
    """Load the real folds CSV, skip if not available."""
    if not FOLDS_PATH.exists():
        pytest.skip("densenet121_folds.csv not yet generated")
    import pandas as pd
    return pd.read_csv(FOLDS_PATH)


class TestRealFolds:
    """Tests using data/manifests/densenet121_folds.csv."""

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_uses_only_train_labels(self, folds_df, fold: int) -> None:
        """Weights are computed from fold != k only."""
        train_df = folds_df[folds_df["fold"] != fold]
        val_df = folds_df[folds_df["fold"] == fold]
        train_labels = train_df["class_id"].tolist()

        # The train set must NOT contain any validation indices
        train_ids = set(train_df["image_id"])
        val_ids = set(val_df["image_id"])
        assert train_ids.isdisjoint(val_ids), "Train and val overlap!"

        weights = compute_balanced_class_weights(train_labels, num_classes=22)
        assert len(weights) == 22

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_22_weights_produced(self, folds_df, fold: int) -> None:
        train_labels = folds_df[folds_df["fold"] != fold]["class_id"].tolist()
        weights = compute_balanced_class_weights(train_labels, num_classes=22)
        assert len(weights) == 22
        assert set(weights.keys()) == set(range(22))

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_no_nan_or_inf(self, folds_df, fold: int) -> None:
        import math
        train_labels = folds_df[folds_df["fold"] != fold]["class_id"].tolist()
        weights = compute_balanced_class_weights(train_labels, num_classes=22)
        for cid, w in weights.items():
            assert not math.isnan(w), f"NaN weight for class {cid}"
            assert not math.isinf(w), f"Inf weight for class {cid}"

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_muscle_heavier_than_oesophagus(self, folds_df, fold: int) -> None:
        """Muscle (small class) must get a higher weight than oesophagus (large class)."""
        train_df = folds_df[folds_df["fold"] != fold]
        train_labels = train_df["class_id"].tolist()

        # Find class_id for muscle and oesophagus
        muscle_id = train_df[train_df["class_name"] == "muscle"]["class_id"].iloc[0]
        oesophagus_id = train_df[train_df["class_name"] == "oesophagus"]["class_id"].iloc[0]

        weights = compute_balanced_class_weights(train_labels, num_classes=22)
        assert weights[muscle_id] > weights[oesophagus_id], (
            f"Fold {fold}: muscle weight ({weights[muscle_id]:.4f}) should be > "
            f"oesophagus weight ({weights[oesophagus_id]:.4f})"
        )

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_all_22_classes_in_train(self, folds_df, fold: int) -> None:
        """Every fold's train set must contain all 22 classes."""
        train_df = folds_df[folds_df["fold"] != fold]
        assert train_df["class_id"].nunique() == 22
