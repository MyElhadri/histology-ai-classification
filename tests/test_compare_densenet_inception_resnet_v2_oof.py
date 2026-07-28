"""Unit tests for DenseNet121 vs InceptionResNetV2 OOF comparison script.

Tests:
- Alignment on image_id + fold + true_label
- Rejection on mismatches, duplicates, missing data
- 50/50 ensemble formula (not optimized on screening)
- Complementarity metrics
- Output files created (comparison_summary.json, per_image_comparison.csv, etc.)
- Abort conditions (class_mapping mismatch, missing probabilities)

No scientific training, no ImageNet download.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NUM_CLASSES = 22
SCREENING_FOLDS = [3, 0, 4]
FOLD_SIZES = {3: 86, 0: 87, 4: 86}

PROB_COLS = [f"prob_{c}" for c in range(NUM_CLASSES)]


def _make_synthetic_oof_df(fold: int, n: int, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic OOF DataFrame for testing."""
    np.random.seed(seed)
    true_labels = np.random.randint(0, NUM_CLASSES, size=n)

    rows = []
    for i in range(n):
        probs = np.random.dirichlet(np.ones(NUM_CLASSES))
        pred = int(np.argmax(probs))
        row = {
            "image_path": f"fold_{fold}/img_{i}.png",
            "image_id": f"f{fold}_img_{i:04d}",
            "fold": fold,
            "true_label": int(true_labels[i]),
            "true_class": f"class_{true_labels[i]}",
            "predicted_label": pred,
            "predicted_class": f"class_{pred}",
            "correct": bool(true_labels[i] == pred),
        }
        for c in range(NUM_CLASSES):
            row[f"prob_{c}"] = float(probs[c])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_fold_csvs(
    base_dir: Path,
    folds: list[int] = SCREENING_FOLDS,
    seed_offset_dense: int = 0,
    seed_offset_irv2: int = 100,
    true_labels_override: dict | None = None,
) -> tuple[Path, Path]:
    """Write synthetic prediction CSVs for DenseNet and InceptionResNetV2."""
    dense_dir = base_dir / "densenet"
    irv2_dir = base_dir / "irv2"
    dense_dir.mkdir(parents=True, exist_ok=True)
    irv2_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        n = FOLD_SIZES[fold]
        dense_df = _make_synthetic_oof_df(fold, n, seed=seed_offset_dense + fold)
        irv2_df = _make_synthetic_oof_df(fold, n, seed=seed_offset_irv2 + fold)

        # Keep same image_id and true_label for alignment
        irv2_df["image_id"] = dense_df["image_id"]
        irv2_df["true_label"] = dense_df["true_label"]
        irv2_df["true_class"] = dense_df["true_class"]

        if true_labels_override:
            for fold_id, labels in true_labels_override.items():
                if fold_id == fold:
                    dense_df["true_label"] = labels
                    irv2_df["true_label"] = labels

        dense_df.to_csv(dense_dir / f"fold_{fold}_oof_predictions.csv", index=False)
        irv2_df.to_csv(irv2_dir / f"fold_{fold}_oof_predictions.csv", index=False)

    return dense_dir, irv2_dir


class TestLoadAndValidate:
    def test_valid_alignment_succeeds(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import load_and_validate_predictions
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        assert len(dense_df) == 259
        assert len(irv2_df) == 259

    def test_missing_dense_file_raises(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import load_and_validate_predictions
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        # Remove one file
        (dense_dir / "fold_3_oof_predictions.csv").unlink()
        with pytest.raises(FileNotFoundError):
            load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)

    def test_missing_irv2_file_raises(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import load_and_validate_predictions
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        (irv2_dir / "fold_0_oof_predictions.csv").unlink()
        with pytest.raises(FileNotFoundError):
            load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)

    def test_true_label_mismatch_raises(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import load_and_validate_predictions
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)

        # Corrupt one irv2 file: change true_labels
        csv_path = irv2_dir / "fold_3_oof_predictions.csv"
        df = pd.read_csv(csv_path)
        df["true_label"] = (df["true_label"] + 1) % NUM_CLASSES
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="true_label mismatch"):
            load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)

    def test_duplicates_in_dense_raise(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import load_and_validate_predictions
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)

        # Duplicate a row in dense fold_3
        csv_path = dense_dir / "fold_3_oof_predictions.csv"
        df = pd.read_csv(csv_path)
        df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate first row
        df_dup.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="[Dd]uplicat"):
            load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)

    def test_missing_prob_cols_raise(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import load_and_validate_predictions
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)

        # Remove prob cols from irv2 fold_3
        csv_path = irv2_dir / "fold_3_oof_predictions.csv"
        df = pd.read_csv(csv_path)
        df_stripped = df.drop(columns=[f"prob_{c}" for c in range(NUM_CLASSES)])
        df_stripped.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing column"):
            load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)


class TestComplementarityMetrics:
    def test_complementarity_metrics_computed(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            compute_complementarity,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        results = compute_complementarity(dense_df, irv2_df)

        assert "images_evaluated" in results
        assert results["images_evaluated"] == 259
        assert "single_model_metrics" in results
        assert "complementarity_breakdown" in results
        assert "simple_ensemble_50_50" in results

    def test_ensemble_weights_not_optimized(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            compute_complementarity,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        results = compute_complementarity(dense_df, irv2_df)
        assert results["simple_ensemble_50_50"]["weights_optimized_on_screening"] is False

    def test_ensemble_50_50_formula_correct(self, tmp_path: Path) -> None:
        """Verify ensemble is simple average, not weighted."""
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            compute_complementarity,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        results = compute_complementarity(dense_df, irv2_df)

        # Verify by computing the ensemble manually
        dense_probs = dense_df[PROB_COLS].values
        irv2_probs = irv2_df[PROB_COLS].values
        ensemble_probs = (dense_probs + irv2_probs) / 2.0
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
        y_true = dense_df["true_label"].values

        from sklearn.metrics import accuracy_score
        expected_ens_acc = float(accuracy_score(y_true, ensemble_pred))
        assert abs(results["simple_ensemble_50_50"]["accuracy"] - expected_ens_acc) < 0.001

    def test_complementarity_breakdown_sums_to_total(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            compute_complementarity,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        results = compute_complementarity(dense_df, irv2_df)

        comp = results["complementarity_breakdown"]
        total = results["images_evaluated"]
        assert (
            comp["both_correct"]
            + comp["both_incorrect"]
            + comp["densenet_only_correct"]
            + comp["inception_resnet_v2_only_correct"]
        ) == total

    def test_scope_warning_present(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            compute_complementarity,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        results = compute_complementarity(dense_df, irv2_df)
        assert "scope_warning" in results


class TestOutputFiles:
    def test_all_output_files_created(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            build_per_image_comparison,
            compute_complementarity,
            generate_complementarity_summary_md,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        output_dir = tmp_path / "comparison_output"
        output_dir.mkdir(parents=True)

        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        results = compute_complementarity(dense_df, irv2_df)
        per_image_df = build_per_image_comparison(dense_df, irv2_df)

        # Save files
        (output_dir / "comparison_summary.json").write_text(json.dumps(results, indent=2))
        per_image_df.to_csv(output_dir / "per_image_comparison.csv", index=False)
        (output_dir / "ensemble_50_50_metrics.json").write_text(
            json.dumps(results["simple_ensemble_50_50"], indent=2)
        )
        generate_complementarity_summary_md(results, output_dir / "complementarity_summary.md")

        assert (output_dir / "comparison_summary.json").is_file()
        assert (output_dir / "per_image_comparison.csv").is_file()
        assert (output_dir / "ensemble_50_50_metrics.json").is_file()
        assert (output_dir / "complementarity_summary.md").is_file()

    def test_per_image_comparison_has_required_columns(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            build_per_image_comparison,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        per_image_df = build_per_image_comparison(dense_df, irv2_df)

        required_cols = [
            "image_id", "fold", "true_label",
            "densenet_pred", "irv2_pred", "ensemble_pred",
            "densenet_correct", "irv2_correct", "ensemble_correct",
            "agreement",
        ]
        for col in required_cols:
            assert col in per_image_df.columns, f"Missing column: {col}"

    def test_per_image_has_259_rows(self, tmp_path: Path) -> None:
        from scripts.compare_densenet_inception_resnet_v2_oof import (
            build_per_image_comparison,
            load_and_validate_predictions,
        )
        dense_dir, irv2_dir = _write_fold_csvs(tmp_path)
        dense_df, irv2_df = load_and_validate_predictions(dense_dir, irv2_dir, SCREENING_FOLDS)
        per_image_df = build_per_image_comparison(dense_df, irv2_df)
        assert len(per_image_df) == 259
