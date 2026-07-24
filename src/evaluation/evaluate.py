"""Evaluation metrics computation.

Computes accuracy, macro F1, weighted F1, confusion matrix, and classification report.
"""

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(
    model: Any,
    dataset: tf.data.Dataset,
    minority_class_ids: list[int] | None = None,
    num_classes: int = 22,
) -> Dict[str, Any]:
    """Evaluate a trained model on a tf.data.Dataset.
    
    Args:
        model: A compiled and trained Keras Model.
        dataset: Validation or test tf.data.Dataset.
        minority_class_ids: List of class IDs considered minority classes.
        num_classes: Number of target classes.
        
    Returns:
        Dictionary containing computed metrics.
    """
    y_true = []
    y_pred = []
    
    cce_metric = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)

    for images, labels_onehot in dataset:
        predictions = model.predict(images, verbose=0)
        cce_metric.update_state(labels_onehot, predictions)
        
        y_true.extend(np.argmax(labels_onehot.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    val_ce_hard = float(cce_metric.result().numpy())
    
    zero_f1_count = int(sum(1 for cid in range(num_classes) if cr.get(str(cid), {}).get("f1-score", 0.0) == 0.0))

    if minority_class_ids:
        minority_f1s = [cr.get(str(cid), {}).get("f1-score", 0.0) for cid in minority_class_ids]
        minority_recalls = [cr.get(str(cid), {}).get("recall", 0.0) for cid in minority_class_ids]
        minority_macro_f1 = float(np.mean(minority_f1s)) if minority_f1s else 0.0
        minority_macro_r = float(np.mean(minority_recalls)) if minority_recalls else 0.0
    else:
        minority_macro_f1 = 0.0
        minority_macro_r = 0.0

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "val_ce_hard": val_ce_hard,
        "minority_macro_f1": minority_macro_f1,
        "minority_macro_recall": minority_macro_r,
        "zero_f1_class_count": zero_f1_count,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }

