"""Builder script for notebooks/colab/inceptionv3_exp_a_complete_training.ipynb using nbformat."""

import json
from pathlib import Path
import nbformat as nbf


def build_inceptionv3_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Introduction Markdown
    c1_md = r"""# InceptionV3 Expérience A — Screening Folds 0, 3 & 4

Ce notebook Google Colab permet d'exécuter l'entraînement, l'évaluation et la génération de rapports pour **InceptionV3 Expérience A (Fair Comparison)**.

## Protocole Scientifique Strict :
- **Architecture** : InceptionV3 pré-entraîné sur ImageNet (`weights="imagenet"`)
- **Taille d'entrée** : $224 \times 224 \times 3$ en `float32` nativement dans $[0, 255]$
- **Preprocessing interne** : Couche sérialisable `tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0)` transformant $[0, 255] \to [-1, 1]$. Aucune division externe par 255.
- **Tête de classification** : Architecture `article_inspired` identique à DenseNet121 D (GlobalAveragePooling2D $\to$ Dense 512 ELU $\to$ BatchNorm $\to$ Dropout 0.30 $\to$ Dense 128 ELU L2=0.01 $\to$ Dense 22 Softmax).
- **Manifeste autoritaire** : `data/manifests/densenet121_folds.csv` (432 images originales, 22 classes, seed 42).
- **Folds de Screening** : Folds **0, 3 et 4** (259 images de validation au total).
- **Augmentation** : Augmentation riche uniquement sur le train set ; images originales brutes pour la validation ; aucune TTA.
- **Stratégie 2-Phase Fine-Tuning** :
  - **Phase 1** : Entraînement de la tête seule (LR = 0.001, backbone gelé).
  - **Phase 2** : Fine-tuning du backbone (LR = $10^{-5}$, toutes les couches `BatchNormalization` du backbone restent strictement non entraînables).
"""
    cells.append(nbf.v4.new_markdown_cell(c1_md))

    # Cell 2: User Config Markdown
    c2_md = r"""## 1. Cellule Unique de Configuration Utilisateur

Tous les paramètres modifiables par l'utilisateur sont centralisés dans cette cellule.
"""
    cells.append(nbf.v4.new_markdown_cell(c2_md))

    # Cell 3: User Config Code
    c3_code = r"""# PARAMÈTRES PRINCIPAUX DE L'EXPÉRIMENTATION INCEPTIONV3
REPO_URL = "https://github.com/MyElhadri/histology-ai-classification.git"
BRANCH = "main"

PROJECT_DIR = "/content/histology-ai-classification"

DRIVE_DATASET = (
    "/content/drive/MyDrive/histology-ai-classification/"
    "data/nuinsseg_human_22_original"
)

LOCAL_DATASET = (
    "/content/histology-ai-classification/"
    "data/raw/nuinsseg_human_22_original"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/histology-results/"
    "inceptionv3-exp-a-screening-folds-0-3-4"
)

CONFIG_PATH = (
    "configs/experiments/"
    "inceptionv3_exp_a_fair_comparison.yaml"
)

FOLDS_TO_RUN = [0, 3, 4]
SEED = 42

RUN_TESTS = True
RUN_TRAINING = True
GENERATE_REPORT = True
SKIP_COMPLETED_FOLDS = True
ALLOW_OVERWRITE = False

print("Configuration Utilisateur InceptionV3 chargée avec succès.")
print(f"Folds de screening sélectionnés : {FOLDS_TO_RUN}")
print(f"Dossier de sauvegarde Drive : {OUTPUT_DIR}")
"""
    cells.append(nbf.v4.new_code_cell(c3_code))

    # Cell 4: GPU Check Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 2. Vérification du GPU et de l'Environnement TensorFlow"))

    # Cell 5: GPU Check Code
    c5_code = r"""import sys
import tensorflow as tf

print(f"Version de Python : {sys.version}")
print(f"Version de TensorFlow : {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("AUCUN GPU DÉTECTÉ ! Veuillez activer un accélérateur GPU dans Colab (Exécution > Modifier le type d'exécution).")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU Détecté et configuré (Memory Growth ON) : {gpu}")

!nvidia-smi
"""
    cells.append(nbf.v4.new_code_cell(c5_code))

    # Cell 6: Drive Mount Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 3. Montage de Google Drive"))

    # Cell 7: Drive Mount Code
    c7_code = r"""import os
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
print(f"Google Drive monté. Dossier de sortie prêt : {output_path}")
"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # Cell 8: Git Clone/Pull Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 4. Clonage ou Synchronisation du Dépôt GitHub"))

    # Cell 9: Git Clone/Pull Code
    c9_code = r"""import os
import subprocess

if not os.path.exists(PROJECT_DIR):
    print(f"Clonage du dépôt depuis {REPO_URL} (branche {BRANCH})...")
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, PROJECT_DIR], check=True)
else:
    print(f"Mise à jour du dépôt existant dans {PROJECT_DIR}...")
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=PROJECT_DIR, check=True)
    subprocess.run(["git", "checkout", BRANCH], cwd=PROJECT_DIR, check=True)
    subprocess.run(["git", "pull", "origin", BRANCH], cwd=PROJECT_DIR, check=True)

os.chdir(PROJECT_DIR)
print(f"Dossier de travail actif : {os.getcwd()}")
!git status -s
!git log -1 --oneline
"""
    cells.append(nbf.v4.new_code_cell(c9_code))

    # Cell 10: Mandatory Files Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 5. Vérification des Fichiers Obligatoires"))

    # Cell 11: Mandatory Files Code
    c11_code = r"""from pathlib import Path

required_files = [
    PROJECT_DIR + "/src/models/inceptionv3.py",
    PROJECT_DIR + "/scripts/train_inceptionv3.py",
    PROJECT_DIR + "/scripts/generate_inceptionv3_screening_report.py",
    PROJECT_DIR + "/scripts/compare_densenet_inceptionv3_oof.py",
    PROJECT_DIR + "/" + CONFIG_PATH,
    PROJECT_DIR + "/data/manifests/densenet121_folds.csv",
    PROJECT_DIR + "/data/manifests/class_mapping.json",
]

missing_files = [f for f in required_files if not Path(f).is_file()]
if missing_files:
    raise FileNotFoundError(f"Fichiers requis manquants : {missing_files}")

print("Tous les fichiers obligatoires du projet sont présents et valides.")
"""
    cells.append(nbf.v4.new_code_cell(c11_code))

    # Cell 12: Dependencies Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 6. Installation des Dépendances et Contrôle de Version"))

    # Cell 13: Dependencies Code
    c13_code = r"""import subprocess
import sys

req_colab = Path(PROJECT_DIR) / "requirements-colab.txt"
req_std = Path(PROJECT_DIR) / "requirements.txt"

target_req = req_colab if req_colab.is_file() else req_std
print(f"Installation des dépendances depuis {target_req}...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(target_req)], check=True)

import numpy as np
import pandas as pd
import sklearn
import yaml

print("\n--- Versions des bibliothèques clés ---")
print(f"NumPy : {np.__version__}")
print(f"Pandas : {pd.__version__}")
print(f"Scikit-Learn : {sklearn.__version__}")
print(f"PyYAML : {yaml.__version__}")
"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # Cell 14: Dataset Preparation Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 7. Préparation et Vérification du Dataset d'Images Originales"))

    # Cell 15: Dataset Preparation Code
    c15_code = r"""import os
from pathlib import Path

drive_ds = Path(DRIVE_DATASET)
local_ds = Path(LOCAL_DATASET)

if not drive_ds.is_dir():
    raise FileNotFoundError(f"Le dataset source Google Drive est introuvable à l'emplacement : {drive_ds}")

local_ds.parent.mkdir(parents=True, exist_ok=True)
if local_ds.is_symlink() or local_ds.exists():
    if local_ds.is_symlink():
        local_ds.unlink()

os.symlink(drive_ds, local_ds)
print(f"Lien symbolique créé : {local_ds} -> {drive_ds}")

valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
image_files = [p for p in local_ds.rglob("*") if p.suffix.lower() in valid_exts]
print(f"Nombre total d'images trouvées dans le dataset : {len(image_files)}")

if len(image_files) != 432:
    raise ValueError(f"Le dataset doit contenir exactement 432 images originales, trouvé : {len(image_files)}")

print("Dataset vérifié avec succès (exactement 432 images originales).")
"""
    cells.append(nbf.v4.new_code_cell(c15_code))

    # Cell 16: Manifest Verification Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 8. Vérification du Manifeste Autoritaire et de la Distribution des Folds"))

    # Cell 17: Manifest Verification Code
    c17_code = r"""import pandas as pd
from pathlib import Path

manifest_path = Path(PROJECT_DIR) / "data/manifests/densenet121_folds.csv"
df_manifest = pd.read_csv(manifest_path)

print(f"Total d'images dans le manifeste autoritaire : {len(df_manifest)}")
assert len(df_manifest) == 432, "Le manifeste doit contenir 432 lignes."
assert df_manifest["image_id"].nunique() == 432, "Tous les image_id doivent être uniques."

fold_counts = df_manifest["fold"].value_counts().sort_index().to_dict()
print("\nRépartition exacte des images par fold :")
for f_id, count in fold_counts.items():
    print(f" - Fold {f_id} : {count} images")

expected_counts = {0: 87, 1: 87, 2: 86, 3: 86, 4: 86}
for f_id, exp_c in expected_counts.items():
    assert fold_counts[f_id] == exp_c, f"Erreur sur le fold {f_id} : attendu {exp_c}, obtenu {fold_counts[f_id]}"

print("\nValidation de la répartition des 5 folds réussie (0:87, 1:87, 2:86, 3:86, 4:86).")
"""
    cells.append(nbf.v4.new_code_cell(c17_code))

    # Cell 18: YAML Inspection Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 9. Inspection et Validation de la Configuration InceptionV3"))

    # Cell 19: YAML Inspection Code
    c19_code = r"""import yaml
from pathlib import Path

yaml_file = Path(PROJECT_DIR) / CONFIG_PATH
with open(yaml_file, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=== CONFIGURATION INCEPTIONV3 EXP A ===")
print(f"Architecture : {cfg['model']['architecture']}")
print(f"Taille d'entrée : {cfg['model']['input_shape']}")
print(f"Nombre de classes : {cfg['model']['num_classes']}")
print(f"Poids d'origine : {cfg['model']['weights']}")
print(f"Tête de classification : {cfg['model']['classifier_head']['type']}")
print(f"Preprocessing méthode : {cfg['preprocessing']['method']}")
print(f"Folds de screening : {cfg['screening']['folds']}")

assert cfg['model']['architecture'] == "InceptionV3"
assert cfg['preprocessing']['external_divide_by_255'] is False
assert cfg['preprocessing']['external_preprocess_input'] is False
assert cfg['screening']['folds'] == [0, 3, 4]
print("\nValidation automatique de la configuration YAML réussie.")
"""
    cells.append(nbf.v4.new_code_cell(c19_code))

    # Cell 20: Pytest Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 10. Exécution des Tests Unitaires Automatisés (Pytest)"))

    # Cell 21: Pytest Code
    c21_code = r"""import subprocess
import sys

if RUN_TESTS:
    print("Lancement de la suite de tests unitaires ciblés InceptionV3...\n")
    test_cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_inceptionv3.py",
        "tests/test_compare_densenet_inceptionv3_oof.py",
        "tests/test_inceptionv3_report.py",
        "-v"
    ]
    res = subprocess.run(test_cmd, cwd=PROJECT_DIR)
    if res.returncode != 0:
        raise RuntimeError("ÉCHEC DES TESTS UNITAIRES ! Corrigez les erreurs avant d'entraîner le modèle.")
    print("\nTous les tests unitaires ont réussi avec succès.")
else:
    print("RUN_TESTS est défini sur False. Étape de test ignorée.")
"""
    cells.append(nbf.v4.new_code_cell(c21_code))

    # Cell 22: Resume & Existing State Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 11. État des Résultats Existants et Reprise après Interruption"))

    # Cell 23: Resume & Existing State Code
    c23_code = r"""from pathlib import Path
import pandas as pd

out_dir = Path(OUTPUT_DIR)
status_rows = []

for f in [0, 1, 2, 3, 4]:
    ckpt = out_dir / "models" / "inceptionv3" / "checkpoints" / f"fold_{f}" / "best_model.keras"
    met = out_dir / "reports" / "inceptionv3" / "metrics" / f"fold_{f}.json"
    oof = out_dir / "reports" / "inceptionv3" / "predictions" / f"fold_{f}_oof_predictions.csv"
    h_json = out_dir / "reports" / "inceptionv3" / "history" / f"fold_{f}_history.json"
    h_csv = out_dir / "reports" / "inceptionv3" / "history" / f"fold_{f}_history.csv"
    cw_json = out_dir / "reports" / "inceptionv3" / "class_weights" / f"fold_{f}_class_weights.json"
    
    is_complete = all(x.is_file() and x.stat().st_size > 0 for x in [ckpt, met, oof, h_json, h_csv, cw_json])
    
    status_rows.append({
        "fold": f,
        "complete": is_complete,
        "checkpoint": ckpt.is_file(),
        "metrics": met.is_file(),
        "oof_predictions": oof.is_file(),
        "history": h_json.is_file(),
        "class_weights": cw_json.is_file(),
    })

df_status = pd.DataFrame(status_rows)
print("=== ÉTAT DES OUTPUTS PAR FOLD SUR GOOGLE DRIVE ===")
print(df_status.to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c23_code))

    # Cell 24: Smoke Test Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 12. Smoke Test de l'Architecture InceptionV3"))

    # Cell 25: Smoke Test Code
    c25_code = r"""import gc
import numpy as np
import tensorflow as tf
from src.models.inceptionv3 import build_inceptionv3, verify_no_double_preprocessing

print("Instanciation d'un modèle InceptionV3 sans poids (weights=None) pour valider les formes...")
smoke_model = build_inceptionv3(
    num_classes=22,
    input_shape=(224, 224, 3),
    weights=None,
    head_config=cfg['model']['classifier_head']
)

assert smoke_model.input_shape == (None, 224, 224, 3)
assert smoke_model.output_shape == (None, 22)
verify_no_double_preprocessing(smoke_model)

dummy_batch = np.random.uniform(0.0, 255.0, size=(2, 224, 224, 3)).astype(np.float32)
probs = smoke_model.predict(dummy_batch, verbose=0)
assert probs.shape == (2, 22)
np.testing.assert_allclose(np.sum(probs, axis=1), [1.0, 1.0], rtol=1e-4)

print("Smoke test réussi : Formes (2, 224, 224, 3) -> (2, 22) et Softmax validés.")

del smoke_model
tf.keras.backend.clear_session()
gc.collect()
"""
    cells.append(nbf.v4.new_code_cell(c25_code))

    # Cell 26: Screening Launch Header Markdown
    cells.append(nbf.v4.new_markdown_cell(r"# 🚀 LANCEMENT DU SCREENING INCEPTIONV3 — FOLDS 0, 3 ET 4"))

    # Cell 27: Fold Execution Loop Code (Real-time unbuffered logging)
    c27_code = r"""import os
import subprocess
import sys
from pathlib import Path

if RUN_TRAINING:
    print(f"Exécution fold par fold des folds {FOLDS_TO_RUN} en temps réel unbuffered...\n")
    
    for fold in FOLDS_TO_RUN:
        print(f"\n========================================================")
        print(f">>> TRAITEMENT ET ENTRAÎNEMENT DU FOLD {fold}")
        print(f"========================================================\n")
        
        cmd_fold = [
            sys.executable, "-u", "scripts/train_inceptionv3.py",
            "--config", CONFIG_PATH,
            "--dataset-dir", LOCAL_DATASET,
            "--output-dir", OUTPUT_DIR,
            "--folds", str(fold)
        ]
        if SKIP_COMPLETED_FOLDS:
            cmd_fold.append("--skip-completed")
        if ALLOW_OVERWRITE:
            cmd_fold.append("--overwrite")
            
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
        
        proc = subprocess.Popen(
            cmd_fold,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_DIR,
            env=env
        )
        
        for line in proc.stdout:
            print(line, end="", flush=True)
            
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ÉCHEC LORS DE L'ENTRAÎNEMENT DU FOLD {fold} (Code d'erreur : {proc.returncode})")
            
    print("\nLancement et exécution de l'entraînement des folds terminés avec succès.")
else:
    print("\nRUN_TRAINING est défini sur False. Entraînement ignoré.")
"""
    cells.append(nbf.v4.new_code_cell(c27_code))

    # Cell 28: Checkpoint Validation Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 14. Validation et Contrôle des Checkpoints Générés"))

    # Cell 29: Checkpoint Validation Code
    c29_code = r"""import gc
import numpy as np
import tensorflow as tf
from pathlib import Path

out_dir = Path(OUTPUT_DIR)

for f in FOLDS_TO_RUN:
    ckpt_file = out_dir / "models" / "inceptionv3" / "checkpoints" / f"fold_{f}" / "best_model.keras"
    if not ckpt_file.is_file():
        print(f"AVERTISSEMENT : Checkpoint manquant pour le fold {f}")
        continue
        
    size_mb = ckpt_file.stat().st_size / (1024 * 1024)
    print(f"Validation du checkpoint Fold {f} ({size_mb:.2f} MB)...")
    
    m = tf.keras.models.load_model(str(ckpt_file), compile=False)
    assert m.output_shape == (None, 22)
    
    test_batch = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    p = m.predict(test_batch, verbose=0)
    assert p.shape == (1, 22)
    np.testing.assert_allclose(np.sum(p), 1.0, rtol=1e-4)
    print(f" -> Checkpoint Fold {f} valide (Softmax Sum = {np.sum(p):.4f})")
    
    del m
    tf.keras.backend.clear_session()
    gc.collect()
"""
    cells.append(nbf.v4.new_code_cell(c29_code))

    # Cell 30: Metrics Table Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 15. Analyse des Métriques du Screening (Folds 0, 3, 4)"))

    # Cell 31: Metrics Table Code
    c31_code = r"""import json
import numpy as np
import pandas as pd
from pathlib import Path

out_dir = Path(OUTPUT_DIR)
metrics_list = []

for f in FOLDS_TO_RUN:
    m_file = out_dir / "reports" / "inceptionv3" / "metrics" / f"fold_{f}.json"
    if m_file.is_file():
        with open(m_file, "r", encoding="utf-8") as fp:
            d = json.load(fp)
            d["fold"] = f
            metrics_list.append(d)

if metrics_list:
    df_metrics = pd.DataFrame(metrics_list)
    print("=== TABLEAU DES MÉTRIQUES INDIVIDUELLES DU SCREENING ===")
    cols_show = ["fold", "accuracy", "macro_f1", "weighted_f1", "correct_samples", "total_samples", "training_duration_seconds"]
    print(df_metrics[[c for c in cols_show if c in df_metrics.columns]].to_string(index=False))
    
    print("\n=== SYNTHÈSE MOYENNE ± ÉCART-TYPE (SCREENING 259 IMAGES) ===")
    print(f"Accuracy Moyenne    : {df_metrics['accuracy'].mean():.4f} ± {df_metrics['accuracy'].std():.4f}")
    print(f"Macro F1 Moyen      : {df_metrics['macro_f1'].mean():.4f} ± {df_metrics['macro_f1'].std():.4f}")
    print(f"Weighted F1 Moyen   : {df_metrics['weighted_f1'].mean():.4f} ± {df_metrics['weighted_f1'].std():.4f}")
else:
    print("Aucun fichier de métriques disponible pour les folds exécutés.")
"""
    cells.append(nbf.v4.new_code_cell(c31_code))

    # Cell 32: OOF Consolidation Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 16. Consolidation des Prédictions Out-Of-Fold (OOF) du Screening"))

    # Cell 33: OOF Consolidation Code
    c33_code = r"""import pandas as pd
from pathlib import Path

out_dir = Path(OUTPUT_DIR)
oof_files = [out_dir / "reports" / "inceptionv3" / "predictions" / f"fold_{f}_oof_predictions.csv" for f in FOLDS_TO_RUN]
existing_oof = [f for f in oof_files if f.is_file()]

if existing_oof:
    dfs = [pd.read_csv(f) for f in existing_oof]
    combined_oof = pd.concat(dfs, ignore_index=True)
    
    print(f"Prédictions OOF consolidées pour {len(combined_oof)} images uniques de validation.")
    assert len(combined_oof) == 259, f"Attendu 259 images pour screening folds 0, 3, 4, obtenu : {len(combined_oof)}"
    
    save_oof_path = out_dir / "reports" / "inceptionv3" / "predictions" / "inceptionv3_screening_oof_predictions.csv"
    save_oof_path.parent.mkdir(parents=True, exist_ok=True)
    combined_oof.to_csv(save_oof_path, index=False)
    print(f"CSV OOF de screening sauvegardé à : {save_oof_path}")
else:
    print("Aucune prédiction OOF trouvée pour la consolidation.")
"""
    cells.append(nbf.v4.new_code_cell(c33_code))

    # Cell 34: Report Generation Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 17. Génération des Matrices de Confusion et Rapports de Présentation"))

    # Cell 35: Report Generation Code
    c35_code = r"""import subprocess
import sys

if GENERATE_REPORT:
    print("Exécution du script de génération des rapports et figures (scripts/generate_inceptionv3_screening_report.py)...")
    cmd_rep = [
        sys.executable, "scripts/generate_inceptionv3_screening_report.py",
        "--output-dir", OUTPUT_DIR
    ]
    env_rep = os.environ.copy()
    env_rep["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env_rep.get("PYTHONPATH", "")
    res = subprocess.run(cmd_rep, check=True, cwd=PROJECT_DIR, env=env_rep)
    print("\nRapports de présentation et matrices 300 DPI générés avec succès.")
"""
    cells.append(nbf.v4.new_code_cell(c35_code))

    # Cell 36: Curves Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 18. Visualisation des Courbes d'Entraînement par Fold"))

    # Cell 37: Curves Code
    c37_code = r"""import json
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path(OUTPUT_DIR)
curves_dir = out_dir / "reports" / "inceptionv3" / "training_curves"
curves_dir.mkdir(parents=True, exist_ok=True)

for f in FOLDS_TO_RUN:
    hist_file = out_dir / "reports" / "inceptionv3" / "history" / f"fold_{f}_history.json"
    if not hist_file.is_file():
        continue
    with open(hist_file, "r", encoding="utf-8") as fp:
        h = json.load(fp)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    axes[0].plot(h.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(h.get("val_accuracy", []), label="Val Accuracy")
    axes[0].set_title(f"InceptionV3 Fold {f} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    
    axes[1].plot(h.get("loss", []), label="Train Loss")
    axes[1].plot(h.get("val_loss", []), label="Val Loss")
    axes[1].set_title(f"InceptionV3 Fold {f} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(curves_dir / f"fold_{f}_curves.png", dpi=300)
    plt.close(fig)
    print(f"Courbes enregistrées pour le Fold {f} -> {curves_dir / f'fold_{f}_curves.png'}")
"""
    cells.append(nbf.v4.new_code_cell(c37_code))

    # Cell 38: DenseNet Comparison Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 19. Comparaison avec DenseNet121 Exp D et Analyse de Complémentarité"))

    # Cell 39: DenseNet Comparison Code
    c39_code = r"""import json
import os
import subprocess
import sys
from pathlib import Path

print("=== RÉFÉRENCE DENSENET121 EXP D (SCREENING FOLDS 0, 3, 4) ===")
print(" - Mean Accuracy   : ~0.8610")
print(" - Mean Macro F1   : ~0.7889")
print(" - Mean Weighted F1: ~0.8538\n")

densenet_oof_csv = Path(PROJECT_DIR) / "artifacts/models/densenet121_exp_d_v1/evaluation/oof_predictions_without_tta.csv"
inc_oof_dir = Path(OUTPUT_DIR) / "reports" / "inceptionv3" / "predictions"

if densenet_oof_csv.is_file() and inc_oof_dir.is_dir():
    print("Exécution de l'analyse de complémentarité OOF (scripts/compare_densenet_inceptionv3_oof.py)...")
    comp_output_json = Path(OUTPUT_DIR) / "reports" / "inceptionv3" / "densenet_inceptionv3_complementarity.json"
    cmd_comp = [
        sys.executable, "scripts/compare_densenet_inceptionv3_oof.py",
        "--densenet-oof", str(densenet_oof_csv),
        "--inception-oof-dir", str(inc_oof_dir),
        "--output", str(comp_output_json)
    ]
    env_comp = os.environ.copy()
    env_comp["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env_comp.get("PYTHONPATH", "")
    res = subprocess.run(cmd_comp, capture_output=True, text=True, cwd=PROJECT_DIR, env=env_comp)
    print(res.stdout)
    if res.returncode == 0 and comp_output_json.is_file():
        with open(comp_output_json, "r", encoding="utf-8") as fp:
            comp_res = json.load(fp)
        print("\n=== Synthèse de Complémentarité & Ensemble 50/50 ===")
        print("Taux de désaccord entre modèles :", comp_res["complementarity_breakdown"]["disagreement_rate"])
        print("Erreurs corrigées par InceptionV3 :", comp_res["complementarity_breakdown"]["inceptionv3_only_correct"])
        print("Erreurs corrigées par DenseNet :", comp_res["complementarity_breakdown"]["densenet_only_correct"])
        print("Accuracy Ensemble 50/50 :", comp_res["simple_ensemble_50_50"]["accuracy"])
else:
    print("Prédictions OOF DenseNet ou InceptionV3 indisponibles. Analyse de complémentarité différée.")
"""
    cells.append(nbf.v4.new_code_cell(c39_code))

    # Cell 40: Final Summary Markdown
    cells.append(nbf.v4.new_markdown_cell(r"## 20. Résumé Final et Verdict d'Exécution"))

    # Cell 41: Final Summary Code
    c41_code = r"""from pathlib import Path

out_dir = Path(OUTPUT_DIR)
screening_folds = [0, 3, 4]

checkpoints_valid = all(
    (out_dir / "models" / "inceptionv3" / "checkpoints" / f"fold_{f}" / "best_model.keras").is_file()
    for f in screening_folds
)
metrics_valid = all(
    (out_dir / "reports" / "inceptionv3" / "metrics" / f"fold_{f}.json").is_file()
    for f in screening_folds
)
oof_valid = all(
    (out_dir / "reports" / "inceptionv3" / "predictions" / f"fold_{f}_oof_predictions.csv").is_file()
    for f in screening_folds
)
summary_file = out_dir / "reports" / "inceptionv3" / "inceptionv3_screening_summary.json"

if checkpoints_valid and metrics_valid and oof_valid and summary_file.is_file():
    print("\n========================================================")
    print("INCEPTIONV3 EXP A SCREENING COMPLETE")
    print("========================================================\n")
    print(f"Tous les artefacts de screening sont enregistrés dans Google Drive : {OUTPUT_DIR}")
else:
    print("\nCertains artefacts de screening sont manquants. Vérifiez les logs ci-dessus.")
"""
    cells.append(nbf.v4.new_code_cell(c41_code))

    # Cell 42: Future Confirmation Note Markdown
    c42_md = r"""## 21. Note sur le Mode Confirmation Futur

Si les critères de décision (Accuracy $\ge 0.84$, Macro F1 $\ge 0.76$ ou complémentarité d'ensemble positive) sont satisfaits après ce screening, vous pourrez entraîner les Folds 1 et 2 pour obtenir l'évaluation complète sur 432 images en modifiant la configuration initiale :

```python
FOLDS_TO_RUN = [1, 2]
```
"""
    cells.append(nbf.v4.new_markdown_cell(c42_md))

    nb.cells = cells

    output_nb_path = Path("notebooks/colab/inceptionv3_exp_a_complete_training.ipynb")
    output_nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook InceptionV3 généré avec succès à l'emplacement : {output_nb_path.resolve()}")
    print(f"Nombre total de cellules : {len(cells)}")


if __name__ == "__main__":
    build_inceptionv3_notebook()
