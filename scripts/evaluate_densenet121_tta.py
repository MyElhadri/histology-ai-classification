import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

from src.models.densenet121 import build_densenet121
from src.data.pipeline import resolve_image_path
from src.data.create_folds import create_folds

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DenseNet121 with TTA")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--results-folds-0-1", type=str, required=True, help="Dir for folds 0,1 checkpoints")
    parser.add_argument("--results-folds-2-3-4", type=str, required=True, help="Dir for folds 2,3,4 checkpoints")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to evaluate")
    return parser.parse_args()

def find_checkpoint(results_dir: Path, fold: int) -> Path:
    """Find exactly one checkpoint for the given fold."""
    # Since checkpoints are usually named with fold in it, or just .keras/.h5
    # Wait, the dir for fold might contain exactly one checkpoint or multiple.
    # The instructions say: "Ne pas deviner le nom des checkpoints. Lever une erreur lorsqu'un checkpoint est absent ou ambigu."
    # A results dir might contain folds in subfolders or directly.
    # Let's search recursively for .keras or .h5 files that match the fold.
    # Usually it's named something like *fold_0*.keras or similar.
    all_checkpoints = list(results_dir.rglob("*.keras")) + list(results_dir.rglob("*.h5"))
    fold_checkpoints = [ckpt for ckpt in all_checkpoints if f"fold_{fold}" in ckpt.name or f"fold{fold}" in ckpt.name]
    
    if len(fold_checkpoints) == 0:
        raise FileNotFoundError(f"No checkpoint found for fold {fold} in {results_dir}")
    elif len(fold_checkpoints) > 1:
        raise ValueError(f"Ambiguous checkpoints found for fold {fold} in {results_dir}: {fold_checkpoints}")
        
    return fold_checkpoints[0]

def make_tta_dataset(file_paths, batch_size=16, image_size=(224, 224)):
    def load_and_tta(file_path):
        image_string = tf.io.read_file(file_path)
        image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, image_size)
        
        view_0 = image
        view_1 = tf.image.flip_left_right(image)
        view_2 = tf.image.flip_up_down(image)
        view_3 = tf.image.flip_up_down(tf.image.flip_left_right(image))
        
        # return (4, 224, 224, 3)
        return tf.stack([view_0, view_1, view_2, view_3], axis=0)
        
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(load_and_tta, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    
    def reshape_for_batch(images):
        s = tf.shape(images)
        return tf.reshape(images, [s[0] * s[1], s[2], s[3], s[4]])
        
    ds = ds.map(reshape_for_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    args = parse_args()
    
    # 1. Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_size = tuple(config["data"]["image_size"])
    num_classes = config["data"]["num_classes"]
    
    # 2. Read dataset manifest/folds
    manifest_path = _PROJECT_ROOT / config["data"]["manifest_path"]
    folds_path = _PROJECT_ROOT / config["data"]["folds_path"]
    seed = config["project"]["seed"]
    
    # Recreate folds to guarantee exact match
    create_folds(manifest_path, folds_path, num_folds=config["validation"]["num_folds"], seed=seed)
    
    df = pd.read_csv(folds_path)
    
    oof_results_no_tta = []
    oof_results_tta = []
    metrics_summary = {}
    
    all_true_labels = []
    all_preds_no_tta = []
    all_preds_tta = []
    
    for fold in args.folds:
        # Determine which results dir to use
        if fold in [0, 1]:
            results_dir = Path(args.results_folds_0_1)
        else:
            results_dir = Path(args.results_folds_2_3_4)
            
        ckpt_path = find_checkpoint(results_dir, fold)
        
        logger.info(f"Fold {fold}: Loading checkpoint {ckpt_path}")
        
        # 3. Build exact architecture
        model = build_densenet121(
            num_classes=num_classes,
            input_shape=(*image_size, 3),
            weights=None, # loaded from checkpoint
            dropout_rate=config["model"]["dropout_rate"],
            head_config=config["model"]["classifier_head"]
        )
        
        # Verify 7 632 854 params and article_inspired
        assert model.count_params() == 7632854, f"Expected 7,632,854 params, got {model.count_params()}"
        assert config["model"]["classifier_head"]["type"] == "article_inspired"
        
        # Load weights
        model.load_weights(ckpt_path)
        
        # Monkeypatch model.fit to ensure it's not called
        def fit_mock(*args, **kwargs):
            raise RuntimeError("model.fit should not be called during evaluation!")
        model.fit = fit_mock
        
        # 4. Get validation subset
        fold_df = df[df["fold"] == fold].copy()
        
        # Check no train images are here (must be strictly disjoint from train folds)
        # This is implicitly true by df[df["fold"] == fold]
        
        file_paths = []
        for p in fold_df["image_path"]:
            file_paths.append(str(resolve_image_path(p, _PROJECT_ROOT)))
            
        # 5. Create TTA dataset
        ds = make_tta_dataset(file_paths, batch_size=config["training"]["batch_size"], image_size=image_size)
        
        # 6. Predict
        logger.info(f"Predicting for fold {fold} ({len(file_paths)} images)...")
        probs_flat = model.predict(ds) # shape: (N*4, 22)
        
        assert probs_flat.shape[1] == 22, "Expected 22 probability values"
        
        probs = probs_flat.reshape(len(file_paths), 4, 22)
        
        # Ensure probs sum to ~1
        sum_probs = np.sum(probs, axis=-1)
        assert np.allclose(sum_probs, 1.0, atol=1e-3), "Probabilities do not sum to 1"
        
        probs_no_tta = probs[:, 0, :]
        probs_tta = probs.mean(axis=1)
        
        # Sum to 1 for TTA average
        assert np.allclose(np.sum(probs_tta, axis=-1), 1.0, atol=1e-3)
        
        preds_no_tta = np.argmax(probs_no_tta, axis=1)
        preds_tta = np.argmax(probs_tta, axis=1)
        
        conf_no_tta = np.max(probs_no_tta, axis=1)
        conf_tta = np.max(probs_tta, axis=1)
        
        y_true = fold_df["class_id"].values
        
        # Metrics without TTA
        acc_no_tta = accuracy_score(y_true, preds_no_tta)
        f1_mac_no_tta = f1_score(y_true, preds_no_tta, average="macro")
        f1_wei_no_tta = f1_score(y_true, preds_no_tta, average="weighted")
        
        # Look for existing fold_X_metrics.json to verify reproducibility
        existing_metrics_paths = list(results_dir.rglob(f"*fold_{fold}*_metrics.json")) + list(results_dir.rglob(f"fold_{fold}_metrics.json")) + list(results_dir.rglob("metrics.json"))
        if existing_metrics_paths:
            try:
                with open(existing_metrics_paths[0], "r") as f:
                    orig_metrics = json.load(f)
                orig_acc = orig_metrics.get("accuracy", orig_metrics.get("val_accuracy", orig_metrics.get("validation_accuracy", -1.0)))
                if orig_acc != -1.0 and abs(orig_acc - acc_no_tta) > 1e-4:
                    raise ValueError(
                        f"CRITICAL REPRODUCIBILITY ERROR on Fold {fold}:\n"
                        f"Recalculated Accuracy without TTA ({acc_no_tta:.6f}) differs from saved original accuracy ({orig_acc:.6f}).\n"
                        f"Aborting evaluation to prevent flawed comparison."
                    )
                logger.info(f"Fold {fold}: Verified accuracy match without TTA ({acc_no_tta:.6f} vs original {orig_acc:.6f})")
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Could not check original metrics JSON for fold {fold}: {e}")
                
        # Metrics with TTA
        acc_tta = accuracy_score(y_true, preds_tta)
        f1_mac_tta = f1_score(y_true, preds_tta, average="macro")
        f1_wei_tta = f1_score(y_true, preds_tta, average="weighted")
        
        logger.info(f"Fold {fold} - Checkpoint: {ckpt_path.name}")
        logger.info(f"Fold {fold} - Images: {len(y_true)}")
        logger.info(f"Fold {fold} - Accuracy sans TTA : {acc_no_tta:.4f}")
        logger.info(f"Fold {fold} - Accuracy avec TTA : {acc_tta:.4f}")
        logger.info(f"Fold {fold} - Macro F1 sans TTA : {f1_mac_no_tta:.4f}")
        logger.info(f"Fold {fold} - Macro F1 avec TTA : {f1_mac_tta:.4f}")
        
        # Save fold json
        fold_metrics = {
            "fold": fold,
            "images": len(y_true),
            "checkpoint": str(ckpt_path),
            "no_tta": {
                "accuracy": float(acc_no_tta),
                "macro_f1": float(f1_mac_no_tta),
                "weighted_f1": float(f1_wei_no_tta),
                "errors": int(np.sum(y_true != preds_no_tta))
            },
            "tta": {
                "accuracy": float(acc_tta),
                "macro_f1": float(f1_mac_tta),
                "weighted_f1": float(f1_wei_tta),
                "errors": int(np.sum(y_true != preds_tta))
            },
            "delta": {
                "accuracy": float(acc_tta - acc_no_tta),
                "macro_f1": float(f1_mac_tta - f1_mac_no_tta),
                "weighted_f1": float(f1_wei_tta - f1_wei_no_tta),
                "correct_predictions": int(np.sum(y_true == preds_tta) - np.sum(y_true == preds_no_tta))
            }
        }
        with open(output_dir / f"fold_{fold}_tta.json", "w") as f:
            json.dump(fold_metrics, f, indent=4)
            
        metrics_summary[f"fold_{fold}"] = fold_metrics
        
        # Append to OOF
        fold_df["prediction_without_tta"] = preds_no_tta
        fold_df["prediction_with_tta"] = preds_tta
        fold_df["confidence_without_tta"] = conf_no_tta
        fold_df["confidence_with_tta"] = conf_tta
        # Assuming true_class_name is already in manifest, else map it.
        if "class_name" in fold_df.columns:
            fold_df.rename(columns={"class_name": "true_class_name", "class_id": "true_class_id"}, inplace=True, errors="ignore")
            
        oof_results_no_tta.append(fold_df[["image_path", "true_class_id", "true_class_name", "prediction_without_tta", "confidence_without_tta", "fold"]])
        oof_results_tta.append(fold_df[["image_path", "true_class_id", "true_class_name", "prediction_with_tta", "confidence_with_tta", "fold"]])
        
        all_true_labels.extend(y_true)
        all_preds_no_tta.extend(preds_no_tta)
        all_preds_tta.extend(preds_tta)

    # Global OOF
    oof_no_tta_df = pd.concat(oof_results_no_tta, ignore_index=True)
    oof_tta_df = pd.concat(oof_results_tta, ignore_index=True)
    
    assert len(oof_no_tta_df) == 432, f"Expected 432 OOF images exactly once, got {len(oof_no_tta_df)}"
    
    oof_no_tta_df.to_csv(output_dir / "oof_predictions_without_tta.csv", index=False)
    oof_tta_df.to_csv(output_dir / "oof_predictions_with_tta.csv", index=False)
    
    # OOF Metrics
    y_true_oof = np.array(all_true_labels)
    preds_no_tta_oof = np.array(all_preds_no_tta)
    preds_tta_oof = np.array(all_preds_tta)
    
    # Without TTA
    oof_acc_no = accuracy_score(y_true_oof, preds_no_tta_oof)
    oof_f1_mac_no = f1_score(y_true_oof, preds_no_tta_oof, average="macro")
    oof_f1_wei_no = f1_score(y_true_oof, preds_no_tta_oof, average="weighted")
    cm_no = confusion_matrix(y_true_oof, preds_no_tta_oof)
    np.save(output_dir / "confusion_matrix_without_tta.npy", cm_no)
    
    # With TTA
    oof_acc_tta = accuracy_score(y_true_oof, preds_tta_oof)
    oof_f1_mac_tta = f1_score(y_true_oof, preds_tta_oof, average="macro")
    oof_f1_wei_tta = f1_score(y_true_oof, preds_tta_oof, average="weighted")
    cm_tta = confusion_matrix(y_true_oof, preds_tta_oof)
    np.save(output_dir / "confusion_matrix_with_tta.npy", cm_tta)
    
    cr_no = classification_report(y_true_oof, preds_no_tta_oof, output_dict=True, zero_division=0)
    cr_tta = classification_report(y_true_oof, preds_tta_oof, output_dict=True, zero_division=0)
    
    zero_f1_classes_no = sum(1 for v in cr_no.values() if isinstance(v, dict) and 'f1-score' in v and v['f1-score'] == 0.0)
    zero_f1_classes_tta = sum(1 for v in cr_tta.values() if isinstance(v, dict) and 'f1-score' in v and v['f1-score'] == 0.0)
    
    summary = {
        "oof_without_tta": {
            "accuracy": float(oof_acc_no),
            "macro_f1": float(oof_f1_mac_no),
            "weighted_f1": float(oof_f1_wei_no),
            "errors": int(np.sum(y_true_oof != preds_no_tta_oof)),
            "zero_f1_classes": int(zero_f1_classes_no)
        },
        "oof_with_tta": {
            "accuracy": float(oof_acc_tta),
            "macro_f1": float(oof_f1_mac_tta),
            "weighted_f1": float(oof_f1_wei_tta),
            "errors": int(np.sum(y_true_oof != preds_tta_oof)),
            "zero_f1_classes": int(zero_f1_classes_tta)
        },
        "delta": {
            "accuracy": float(oof_acc_tta - oof_acc_no),
            "macro_f1": float(oof_f1_mac_tta - oof_f1_mac_no),
            "weighted_f1": float(oof_f1_wei_tta - oof_f1_wei_no),
            "correct_predictions": int(np.sum(y_true_oof == preds_tta_oof) - np.sum(y_true_oof == preds_no_tta_oof))
        },
        "folds": metrics_summary
    }
    
    with open(output_dir / "tta_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n============================================================")
    print("DÉCISION FINALE")
    print("============================================================")
    if oof_f1_mac_tta > oof_f1_mac_no and oof_acc_tta >= oof_acc_no:
        print("D + TTA SELECTED")
    else:
        print("D WITHOUT TTA SELECTED")
    print("============================================================")

if __name__ == "__main__":
    main()
