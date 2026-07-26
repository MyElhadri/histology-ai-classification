import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def main():
    # Setup paths relative to script location
    scripts_dir = Path(__file__).resolve().parent
    registry_root = scripts_dir.parent
    
    # Project root is 5 levels up from the script:
    # artifacts/models/densenet121_exp_d_v1/scripts/generate_presentation_report.py
    # -> scripts -> densenet121_exp_d_v1 -> models -> artifacts -> histology-ai-classification
    project_root = registry_root.parent.parent.parent
    
    evaluation_dir = registry_root / "evaluation"
    presentation_dir = registry_root / "presentation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = registry_root / "manifests" / "densenet121_folds.csv"
    class_mapping_path = registry_root / "metadata" / "class_mapping.json"
    oof_predictions_path = evaluation_dir / "oof_predictions_without_tta.csv"
    
    print("========== ÉTAPE 1: AUDIT ET CHARGEMENT DES PRÉDICTIONS OOF ==========")
    
    # Check if oof predictions exist and match specifications
    use_existing = False
    if oof_predictions_path.is_file():
        try:
            df_oof = pd.read_csv(oof_predictions_path)
            required_cols = {"image_path", "y_true", "y_pred", "fold"}
            if (
                len(df_oof) == 432
                and len(df_oof["image_path"].unique()) == 432
                and required_cols.issubset(df_oof.columns)
            ):
                use_existing = True
                print("INFO: Found valid existing OOF predictions file. Using it directly.")
        except Exception as e:
            print(f"WARNING: Failed to read existing OOF predictions: {e}")
            
    if not use_existing:
        print("INFO: Generating OOF predictions from fold checkpoints...")
        import tensorflow as tf
        
        # Load manifest
        df_manifest = pd.read_csv(manifest_path)
        
        # Verify sizes and counts
        total_rows = len(df_manifest)
        counts = df_manifest["fold"].value_counts().to_dict()
        assert total_rows == 432, f"Expected 432 images, got {total_rows}"
        assert counts.get(0) == 87, f"Expected 87 images for fold 0, got {counts.get(0)}"
        assert counts.get(1) == 87, f"Expected 87 images for fold 1, got {counts.get(1)}"
        assert counts.get(2) == 86, f"Expected 86 images for fold 2, got {counts.get(2)}"
        assert counts.get(3) == 86, f"Expected 86 images for fold 3, got {counts.get(3)}"
        assert counts.get(4) == 86, f"Expected 86 images for fold 4, got {counts.get(4)}"
        
        # Ensure no duplicates
        assert len(df_manifest["image_path"].unique()) == 432, "Duplicate images found in manifest!"
        
        oof_records = []
        
        # Predict fold by fold
        for fold in range(5):
            ckpt_path = registry_root / "checkpoints" / f"fold_{fold}" / "best_model.keras"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing fold {fold} checkpoint at {ckpt_path}")
                
            print(f"INFO: Running inference for fold {fold}...")
            model = tf.keras.models.load_model(ckpt_path, compile=False)
            
            fold_df = df_manifest[df_manifest["fold"] == fold].copy()
            file_paths = []
            for raw_path in fold_df["image_path"].values:
                # Resolve path relative to project root
                p = Path(raw_path)
                if p.is_absolute():
                    resolved = p
                else:
                    resolved = (project_root / p).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f"Image not found: {resolved}")
                file_paths.append(str(resolved))
                
            # Create tf dataset
            def load_and_resize(file_path):
                img_str = tf.io.read_file(file_path)
                img = tf.image.decode_image(img_str, channels=3, expand_animations=False)
                img = tf.cast(img, tf.float32)
                img = tf.image.resize(img, (224, 224))
                return img
                
            ds = tf.data.Dataset.from_tensor_slices(file_paths)
            ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
            # Batch size 16 as in training config
            ds = ds.batch(16)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            # Predict
            preds = model.predict(ds)
            y_pred = np.argmax(preds, axis=1)
            y_true = fold_df["class_id"].values
            
            # Store records
            for idx, row in enumerate(fold_df.itertuples()):
                oof_records.append({
                    "image_path": row.image_path,
                    "y_true": int(y_true[idx]),
                    "y_pred": int(y_pred[idx]),
                    "fold": fold
                })
                
            # Free memory
            tf.keras.backend.clear_session()
            
        df_oof = pd.DataFrame(oof_records)
        df_oof.to_csv(oof_predictions_path, index=False)
        print(f"INFO: Saved recreated OOF predictions to {oof_predictions_path}")
        
    # Get values as numpy arrays
    y_true_all = df_oof["y_true"].values
    y_pred_all = df_oof["y_pred"].values
    
    correct_preds = int(np.sum(y_true_all == y_pred_all))
    total_preds = len(y_true_all)
    accuracy_oof = accuracy_score(y_true_all, y_pred_all)
    
    print(f"INFO: Correct predictions: {correct_preds}/{total_preds}")
    print(f"INFO: Accuracy OOF: {accuracy_oof:.4f}")
    
    print("\n========== ÉTAPE 4: VÉRIFICATION STRICTE ==========")
    # Exact verification check
    if correct_preds != 382:
        raise ValueError(f"CRITICAL: Number of correct predictions ({correct_preds}) does not match expected (382)!")
    if abs(accuracy_oof - 0.8843) > 1e-4:
        raise ValueError(f"CRITICAL: OOF Accuracy ({accuracy_oof:.6f}) does not match expected (~0.8843)!")
    print("Verification passed! Metrics exactly match official targets.")

    print("\n========== ÉTAPE 2: GÉNÉRATION DU DOSSIER DE PRÉSENTATION ==========")
    
    # Load class mapping
    with open(class_mapping_path, "r") as f:
        class_map = json.load(f)
    classes = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Helper to plot matrices
    def plot_matrix(matrix, title, filepath, is_normalized):
        plt.figure(figsize=(16, 14))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=16, pad=20)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=10)
        plt.yticks(tick_marks, classes, fontsize=10)
        
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if is_normalized:
                    # Ne pas afficher toutes les valeurs si trop petit ou nul (garder propre)
                    if val >= 0.01:
                        plt.text(j, i, f"{val*100:.0f}%",
                                 horizontalalignment="center",
                                 color="white" if val > thresh else "black",
                                 fontsize=8)
                else:
                    if val > 0:
                        plt.text(j, i, str(int(val)),
                                 horizontalalignment="center",
                                 color="white" if val > thresh else "black",
                                 fontsize=8)
                                 
        plt.ylabel('True Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        
    # 1. Plot raw confusion matrix
    raw_png_path = presentation_dir / "confusion_matrix_counts.png"
    plot_matrix(cm, "Matrice de confusion - Nombres Bruts (OOF)", raw_png_path, is_normalized=False)
    print(f"Created {raw_png_path.name}")
    
    # 2. Plot normalized confusion matrix
    norm_png_path = presentation_dir / "confusion_matrix_normalized.png"
    plot_matrix(cm_norm, "Matrice de confusion - Pourcentages Normalisés (OOF)", norm_png_path, is_normalized=True)
    print(f"Created {norm_png_path.name}")
    
    # 3. CSV confusion matrix raw
    df_cm_raw = pd.DataFrame(cm, index=classes, columns=classes)
    raw_csv_path = presentation_dir / "confusion_matrix_counts.csv"
    df_cm_raw.to_csv(raw_csv_path)
    print(f"Created {raw_csv_path.name}")
    
    # 4. CSV confusion matrix normalized
    df_cm_norm = pd.DataFrame(cm_norm, index=classes, columns=classes)
    norm_csv_path = presentation_dir / "confusion_matrix_normalized.csv"
    df_cm_norm.to_csv(norm_csv_path)
    print(f"Created {norm_csv_path.name}")
    
    # Calculate classification report metrics
    report_dict = classification_report(y_true_all, y_pred_all, target_names=classes, output_dict=True, zero_division=0)
    
    # 5. CSV classification report
    report_records = []
    for cls_name in classes:
        cls_metrics = report_dict[cls_name]
        report_records.append({
            "class_name": cls_name,
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1_score": cls_metrics["f1-score"],
            "support": int(cls_metrics["support"])
        })
    df_report = pd.DataFrame(report_records)
    report_csv_path = presentation_dir / "classification_report.csv"
    df_report.to_csv(report_csv_path, index=False)
    print(f"Created {report_csv_path.name}")
    
    # 6. JSON classification report
    report_json_path = presentation_dir / "classification_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Created {report_json_path.name}")
    
    # 7. Overall metrics JSON
    # compute macro / weighted averages
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="macro", zero_division=0)
    prec_wei, rec_wei, f1_wei, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="weighted", zero_division=0)
    
    overall = {
        "accuracy_oof": float(accuracy_oof),
        "macro_precision": float(prec_mac),
        "macro_recall": float(rec_mac),
        "macro_f1": float(f1_mac),
        "weighted_precision": float(prec_wei),
        "weighted_recall": float(rec_wei),
        "weighted_f1": float(f1_wei),
        "correct_predictions": correct_preds,
        "total_predictions": total_preds
    }
    overall_json_path = presentation_dir / "overall_metrics.json"
    with open(overall_json_path, "w") as f:
        json.dump(overall, f, indent=4)
    print(f"Created {overall_json_path.name}")
    
    # 8. Top confusions CSV (top confusions, hors diagonale)
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_class": classes[i],
                    "pred_class": classes[j],
                    "count": int(cm[i, j])
                })
    df_conf = pd.DataFrame(confusions)
    if not df_conf.empty:
        df_conf = df_conf.sort_values(by="count", ascending=False)
    else:
        df_conf = pd.DataFrame(columns=["true_class", "pred_class", "count"])
    top_conf_path = presentation_dir / "top_confusions.csv"
    df_conf.to_csv(top_conf_path, index=False)
    print(f"Created {top_conf_path.name}")
    
    # 9. Model statistics JSON (parameters & weights size)
    # Checkpoint sizes
    ckpt_sizes = {}
    total_ckpt_size = 0
    for fold in range(5):
        ckpt_path = registry_root / "checkpoints" / f"fold_{fold}" / "best_model.keras"
        sz = ckpt_path.stat().st_size
        ckpt_sizes[f"fold_{fold}"] = sz
        total_ckpt_size += sz
        
    # We can load model structure from fold_0 to get TF variables count
    import tensorflow as tf
    fold0_path = registry_root / "checkpoints" / "fold_0" / "best_model.keras"
    model = tf.keras.models.load_model(fold0_path, compile=False)
    
    total_params = int(model.count_params())
    trainable_params = int(sum(v.shape.num_elements() for v in model.trainable_variables))
    non_trainable_params = int(sum(v.shape.num_elements() for v in model.non_trainable_variables))
    
    stats = {
        "model_name": "DenseNet121",
        "input_shape": [224, 224, 3],
        "output_shape": [None, 22],
        "number_of_classes": 22,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "checkpoint_sizes": ckpt_sizes,
        "total_checkpoints_size_bytes": total_ckpt_size,
        "tensorflow_version": tf.__version__,
        "selected_inference": "without_tta"
    }
    stats_json_path = presentation_dir / "model_statistics.json"
    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Created {stats_json_path.name}")
    
    # 10. Model summary txt
    summary_path = presentation_dir / "model_summary.txt"
    # redirect summary output to file
    with open(summary_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Created {summary_path.name}")
    
    # 11. Presentation Summary MD (résumé court en français)
    summary_md_content = """# Résumé de la Présentation Scientifique - DenseNet121 (Expérience D)

Ce document résume les caractéristiques et les performances du meilleur modèle d'apprentissage profond DenseNet121 obtenu au cours de l'évaluation finale.

## Caractéristiques du Modèle & Protocole
- **Architecture de base** : DenseNet121 pré-entraîné sur ImageNet.
- **Taille d'entrée** : 224x224 RGB.
- **Nombre de classes** : 22 classes de tissus histologiques.
- **Dataset d'évaluation** : 432 images originales uniquement (aucune image augmentée ou dupliquée n'a été insérée dans l'évaluation).
- **Protocole de validation** : Validation croisée stratifiée en 5 folds.
- **Régulation de l'apprentissage** :
  - **Augmentation** : Augmentation riche uniquement en ligne (online) pendant l'entraînement.
  - **Tête de classification (Classifier Head)** : Spécifique *article-inspired* comprenant une couche dense de 512 unités (activation ELU), une normalisation par batch, un dropout (rate = 0.30) et une seconde couche dense de 128 unités.
  - **Fine-tuning** : Un fine-tuning complet de toutes les couches du backbone a été réalisé, tout en conservant les couches BatchNormalization du backbone gelées.
- **Inférence** : Inférence classique **sans TTA** (Test-Time Augmentation rejetée car n'apportant pas d'amélioration).

## Performances Globales Out-of-Fold (OOF)
- **Accuracy OOF** : **88,43 %** (382 prédictions correctes sur 432 images).
- **Macro F1 moyen** : **82,80 %**.
- **Weighted F1 moyen** : **87,94 %**.
"""
    summary_md_path = presentation_dir / "presentation_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md_content)
    print(f"Created {summary_md_path.name}")
    
    print("\n============================================================")
    print("DENSENET121 EXP D PRESENTATION REPORT READY")
    print("============================================================")
    
    # List files created in presentation/
    print("Files generated in presentation directory:")
    generated_files = sorted(os.listdir(presentation_dir))
    for gf in generated_files:
        print(f"  - presentation/{gf}")

if __name__ == "__main__":
    main()
