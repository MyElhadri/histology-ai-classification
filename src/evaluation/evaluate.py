"""Evaluation metrics computation.

Computes accuracy, macro F1, weighted F1, confusion matrix, and classification report.
"""

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_model(model: Any, dataset: tf.data.Dataset) -> Dict[str, Any]:
    """Evaluate a trained model on a tf.data.Dataset.
    
    Args:
        model: A compiled and trained Keras Model.
        dataset: Validation or test tf.data.Dataset.
        
    Returns:
        Dictionary containing computed metrics.
    """
    y_true = []
    y_pred = []
    
    # Iterate over the dataset and collect true/predicted labels
    for images, labels_onehot in dataset:
        predictions = model.predict(images, verbose=0)
        
        y_true.extend(np.argmax(labels_onehot.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        
    # Compute metrics
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }
