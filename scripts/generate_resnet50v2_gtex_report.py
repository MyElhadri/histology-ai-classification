"""Report generator for ResNet50V2 on GTEx dataset.

Uses Matplotlib (Agg backend) to generate confusion matrices and training curves.
Outputs a markdown summary and JSON statistics.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], output_path: Path, title: str, normalize: bool = False) -> None:
    cm_arr = np.array(cm, dtype=np.float32)
    if normalize:
        cm_arr = cm_arr / cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.nan_to_num(cm_arr)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else ".0f"
    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            if normalize and val == 0:
                continue
            text = format(val, fmt)
            ax.text(j, i, text, ha="center", va="center", color="white" if val > thresh else "black")

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_cm_csv(cm: list[list[int]], class_names: list[str], output_path: Path, normalize: bool = False) -> None:
    cm_arr = np.array(cm, dtype=np.float32)
    if normalize:
        cm_arr = cm_arr / cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.nan_to_num(cm_arr)
    df = pd.DataFrame(cm_arr, index=class_names, columns=class_names)
    df.to_csv(output_path)


def process_split_report(split: str, results_dir: Path, class_names: list[str]) -> None:
    metrics_path = results_dir / f"{split}_metrics.json"
    if not metrics_path.exists():
        logger.warning(f"No {split} metrics found at {metrics_path}. Skipping.")
        return
        
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Confusion Matrix PNG and CSV (Counts)
    cm = metrics["confusion_matrix"]
    plot_confusion_matrix(cm, class_names, results_dir / f"{split}_confusion_matrix_counts.png", f"Confusion Matrix (Counts) - {split.capitalize()}", normalize=False)
    save_cm_csv(cm, class_names, results_dir / f"{split}_confusion_matrix_counts.csv", normalize=False)

    # 2. Confusion Matrix PNG and CSV (Normalized)
    plot_confusion_matrix(cm, class_names, results_dir / f"{split}_confusion_matrix_normalized.png", f"Confusion Matrix (Normalized) - {split.capitalize()}", normalize=True)
    save_cm_csv(cm, class_names, results_dir / f"{split}_confusion_matrix_normalized.csv", normalize=True)

    # 3. Classification Report CSV
    cr = metrics["classification_report"]
    # Remove aggregate rows for dataframe creation if they exist
    rows = []
    for k, v in cr.items():
        if isinstance(v, dict):
            v["class"] = k
            rows.append(v)
    cr_df = pd.DataFrame(rows)
    cr_df.to_csv(results_dir / f"{split}_classification_report.csv", index=False)

    # 4. Top Confusions CSV
    top_conf_df = pd.DataFrame(metrics["top_confusions"])
    top_conf_df.to_csv(results_dir / f"{split}_top_confusions.csv", index=False)

    # 5. Markdown Presentation Summary
    md_path = results_dir / f"{split}_presentation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ResNet50V2 GTEx {split.capitalize()} Report\n\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
        f.write(f"- **Balanced Accuracy:** {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"- **Macro F1:** {metrics['macro_f1']:.4f}\n")
        f.write(f"- **Cohen's Kappa:** {metrics['cohens_kappa']:.4f}\n")
        
    logger.info(f"Generated reports for split '{split}'.")


def plot_training_curves(results_dir: Path) -> None:
    history_dir = results_dir / "history"
    if not history_dir.exists():
        logger.warning(f"History dir not found at {history_dir}. Skipping curves.")
        return
        
    all_dfs = []
    for p in [1, 2, 3]:
        csv_path = history_dir / f"phase_{p}_training.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["phase"] = p
            all_dfs.append(df)
            
    if not all_dfs:
        return
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Plot accuracy and loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    axes[0].plot(full_df["accuracy"], label="Train Acc")
    if "val_accuracy" in full_df.columns:
        axes[0].plot(full_df["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    
    axes[1].plot(full_df["loss"], label="Train Loss")
    if "val_loss" in full_df.columns:
        axes[1].plot(full_df["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(str(results_dir / "training_curves.png"))
    plt.close(fig)
    logger.info("Generated training curves.")


def generate_model_statistics(results_dir: Path) -> None:
    stats = {
        "framework": "TensorFlow/Keras",
        "architecture": "ResNet50V2",
        "num_classes": 11,
    }
    
    # In a real scenario we could load the model and count parameters.
    # For now we write a placeholder struct. The actual training script saves history.
    with open(results_dir / "model_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    args = parser.parse_args()

    from src.data.gtex_integrity import parse_gtex_class_mapping
    with open(args.class_mapping, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
        
    parsed = parse_gtex_class_mapping(raw_mapping)
    class_names = parsed["classes"]

    process_split_report("validation", args.results_dir, class_names)
    process_split_report("test", args.results_dir, class_names)
    
    plot_training_curves(args.results_dir)
    generate_model_statistics(args.results_dir)

if __name__ == "__main__":
    main()
