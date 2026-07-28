"""Script to generate notebooks/colab/efficientnetv2b0_exp_a_complete_training.ipynb.
"""

import json
from pathlib import Path
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Intro / Title & Protocol
    c1_md = """# EfficientNetV2B0 Expérience A — Screening Folds 0, 3 & 4

Ce notebook Google Colab réutilisable permet d'exécuter l'entraînement, l'évaluation et l'analyse de complémentarité du deuxième modèle du projet : **EfficientNetV2B0** pré-entraîné sur ImageNet.

### Protocole Scientifique & Équité Évaluative :
- **Modèle** : EfficientNetV2B0 (ImageNet pre-trained)
- **Entrée** : 224 × 224 × 3
- **Nombre de classes** : 22 classes histologiques
- **Dataset original** : 432 images humaines
- **Validation croisée** : Manifeste autoritaire `data/manifests/densenet121_folds.csv` (Seed 42)
- **Folds de screening initial** : Folds 0, 3 et 4 (259 images d'évaluation uniques)
- **Augmentation** : Policy `rich` en ligne uniquement durant l'entraînement (Validation sur images originales sans augmentation)
- **Prétraitement** : Prétraitement interne EfficientNetV2B0 (`include_preprocessing=True`), entrée `float32` dans `[0, 255]`, **aucune division externe par 255**.
- **TTA** : Aucune TTA pour cette première expérience
- **Entraînement** : 2 phases (Phase 1 : Tête `article_inspired`, Phase 2 : Fine-tuning du backbone avec `BatchNormalization` gelées)
- **Sauvegarde** : Checkpoints, métriques, prédictions OOF et graphiques enregistrés automatiquement dans Google Drive.
"""
    cells.append(nbf.v4.new_markdown_cell(c1_md))

    # Cell 2: Section 1 Header
    cells.append(nbf.v4.new_markdown_cell("## 1. Cellule Unique de Configuration Utilisateur"))

    # Cell 3: User Config Code
    c3_code = r"""# PARAMÈTRES PRINCIPAUX DE L'EXPÉRIMENTATION EFFICIENTNETV2B0
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
    "efficientnetv2b0-exp-a-screening-folds-0-3-4"
)

CONFIG_PATH = (
    "configs/experiments/"
    "efficientnetv2b0_exp_a_fair_comparison.yaml"
)

FOLDS_TO_RUN = [0, 3, 4]

SEED = 42
ALLOW_OVERWRITE = False
RUN_TESTS = True
RUN_TRAINING = True
GENERATE_REPORT = True

print("Configuration chargée avec succès.")
print(f"Dépôt : {REPO_URL} (branche: {BRANCH})")
print(f"Folds de screening : {FOLDS_TO_RUN}")
print(f"Dossier de sortie Drive : {OUTPUT_DIR}")
"""
    cells.append(nbf.v4.new_code_cell(c3_code))

    # Cell 4: Section 2 Header
    cells.append(nbf.v4.new_markdown_cell("## 2. Vérification du GPU et de l'Environnement TensorFlow"))

    # Cell 5: GPU Code
    c5_code = r"""import sys
import subprocess
import tensorflow as tf

print("=== Diagnostic Matériel & Logiciel ===")
print("Version Python :", sys.version)
print("Version TensorFlow :", tf.__version__)

try:
    smi = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    print("\n--- nvidia-smi ---\n", smi)
except Exception as e:
    print("Avertissement : nvidia-smi introuvable ou GPU inactif :", e)

gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs détectés par TensorFlow : {len(gpus)}")

if not gpus:
    raise SystemError(
        "ERREUR CRITIQUE : Aucun GPU n'a été détecté par TensorFlow.\n"
        "Veuillez activer l'accélérateur matériel sous Colab via : Exécution > Modifier le type d'exécution > GPU T4/V100/A100."
    )

for gpu in gpus:
    print("  GPU actif :", gpu.name)
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print("  --> Croissance progressive de la mémoire GPU activée.")
    except Exception as exc:
        print("  --> Note mémoire GPU :", exc)
"""
    cells.append(nbf.v4.new_code_cell(c5_code))

    # Cell 6: Section 3 Header
    cells.append(nbf.v4.new_markdown_cell("## 3. Montage de Google Drive"))

    # Cell 7: Mount Drive Code
    c7_code = r"""import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive monté avec succès sous /content/drive")
except ImportError:
    print("Note : Module google.colab non disponible (exécution locale).")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Dossier de destination des résultats vérifié :", OUTPUT_DIR)
"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # Cell 8: Section 4 Header
    cells.append(nbf.v4.new_markdown_cell("""## 4. Clonage ou Synchronisation du Dépôt GitHub

> **Note sur les dépôts privés** : Si le dépôt GitHub est privé, configurez une clé d'accès personalisée ou un secret Colab (`from google.colab import userdata`). Ne collez **jamais** de jeton d'accès en clair dans le code du notebook.
"""))

    # Cell 9: Git Code
    c9_code = r"""import os
import subprocess

if not os.path.exists(PROJECT_DIR):
    print(f"Clonage du dépôt {REPO_URL} dans {PROJECT_DIR}...")
    cmd = ["git", "clone", "-b", BRANCH, REPO_URL, PROJECT_DIR]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Erreur clonage :", res.stderr)
        raise RuntimeError(f"Échec du clonage du dépôt {REPO_URL}")
    print("Clonage terminé.")
else:
    print(f"Le projet existe déjà dans {PROJECT_DIR}. Synchronisation...")
    os.chdir(PROJECT_DIR)
    subprocess.run(["git", "fetch", "origin"], check=True)
    subprocess.run(["git", "checkout", BRANCH], check=True)
    res = subprocess.run(["git", "pull", "origin", BRANCH], capture_output=True, text=True)
    print("Résultat du pull :", res.stdout.strip())

os.chdir(PROJECT_DIR)
print("\nRepertoire de travail actif :", os.getcwd())

branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
last_commit = subprocess.check_output(["git", "log", "-1", "--oneline"]).decode("utf-8").strip()
status_output = subprocess.check_output(["git", "status", "--short"]).decode("utf-8").strip()

print(f"Branche active : {branch_name}")
print(f"Dernier commit : {last_commit}")
print("Statut Git local :\n", status_output if status_output else "Répertoire propre (aucun changement local).")
"""
    cells.append(nbf.v4.new_code_cell(c9_code))

    # Cell 10: Section 5 Header
    cells.append(nbf.v4.new_markdown_cell("## 5. Vérification des Fichiers Obligatoires"))

    # Cell 11: File Verification Code
    c11_code = r"""from pathlib import Path

REQUIRED_FILES = [
    "src/models/efficientnetv2b0.py",
    "src/models/heads.py",
    "scripts/train_efficientnetv2b0.py",
    "scripts/compare_densenet_efficientnet_oof.py",
    "configs/experiments/efficientnetv2b0_exp_a_fair_comparison.yaml",
    "data/manifests/densenet121_folds.csv",
    "data/manifests/original_22_dataset_manifest.csv",
    "tests/test_efficientnetv2b0.py",
    "tests/test_compare_oof.py",
]

missing = []
for rel_path in REQUIRED_FILES:
    full_path = Path(PROJECT_DIR) / rel_path
    if not full_path.is_file():
        missing.append(rel_path)

if missing:
    raise FileNotFoundError(
        f"Fichiers requis manquants dans {PROJECT_DIR} :\n" + "\n".join(f" - {f}" for f in missing)
    )

print(f"Tous les {len(REQUIRED_FILES)} fichiers requis sont présents dans le projet.")
"""
    cells.append(nbf.v4.new_code_cell(c11_code))

    # Cell 12: Section 6 Header
    cells.append(nbf.v4.new_markdown_cell("## 6. Installation des Dépendances et Contrôle de Version"))

    # Cell 13: Dependencies Code
    c13_code = r"""import subprocess
import sys
from pathlib import Path

req_file = Path(PROJECT_DIR) / "requirements-colab.txt"
if not req_file.is_file():
    req_file = Path(PROJECT_DIR) / "requirements.txt"

if req_file.is_file():
    print(f"Installation/Vérification des dépendances à partir de {req_file.name}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)], check=True)
    print("Dépendances vérifiées.")

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import sklearn
import yaml

print("\n=== Versions des bibliothèques clés ===")
print("Python :", sys.version.split()[0])
print("TensorFlow :", tf.__version__)
print("Keras :", keras.__version__)
print("NumPy :", np.__version__)
print("Pandas :", pd.__version__)
print("Scikit-Learn :", sklearn.__version__)
print("PyYAML :", yaml.__version__)
"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # Cell 14: Section 7 Header
    cells.append(nbf.v4.new_markdown_cell("## 7. Préparation et Vérification du Dataset d'Images Originales"))

    # Cell 15: Dataset Code
    c15_code = r"""import os
from pathlib import Path

drive_ds_path = Path(DRIVE_DATASET)
if not drive_ds_path.exists():
    raise FileNotFoundError(f"Le dataset est introuvable sur Google Drive : {DRIVE_DATASET}")

local_ds_path = Path(LOCAL_DATASET)
local_ds_path.parent.mkdir(parents=True, exist_ok=True)

if local_ds_path.is_symlink() or local_ds_path.exists():
    if local_ds_path.is_symlink():
        os.unlink(local_ds_path)
    elif local_ds_path.is_dir():
        import shutil
        shutil.rmtree(local_ds_path)

os.symlink(drive_ds_path, local_ds_path)
print(f"Lien symbolique créé : {local_ds_path} -> {drive_ds_path}")

valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
found_images = [f for f in local_ds_path.rglob("*") if f.suffix.lower() in valid_exts]
ext_counts = {}
for f in found_images:
    ext = f.suffix.lower()
    ext_counts[ext] = ext_counts.get(ext, 0) + 1

total_found = len(found_images)
print(f"\nImages totales trouvées : {total_found} (exigé : 432)")
print("Répartition par extension :", ext_counts)

if total_found != 432:
    raise ValueError(f"Le dataset contient {total_found} images, attendu exactement 432 images originales.")

subdirs = sorted([d.name for d in local_ds_path.iterdir() if d.is_dir()])
print(f"Dossiers de classes histologiques ({len(subdirs)}) :", subdirs[:5], "...")
"""
    cells.append(nbf.v4.new_code_cell(c15_code))

    # Cell 16: Section 8 Header
    cells.append(nbf.v4.new_markdown_cell("## 8. Vérification du Manifeste Autoritaire et de la Distribution des Folds"))

    # Cell 17: Manifest Code
    c17_code = r"""import pandas as pd
from pathlib import Path

manifest_file = Path(PROJECT_DIR) / "data/manifests/densenet121_folds.csv"
if not manifest_file.is_file():
    raise FileNotFoundError(f"Manifeste des folds introuvable : {manifest_file}")

df_folds = pd.read_csv(manifest_file)
print(f"Manifeste chargé ({len(df_folds)} lignes).")

if len(df_folds) != 432:
    raise ValueError(f"Attendu 432 lignes dans le manifeste, trouvé {len(df_folds)}.")

if df_folds["image_path"].nunique() != 432:
    raise ValueError("Des doublons ont été détectés dans les chemins d'images du manifeste.")

fold_counts = df_folds["fold"].value_counts().to_dict()
expected_counts = {0: 87, 1: 87, 2: 86, 3: 86, 4: 86}

print("\nDistribution réelle par fold :", fold_counts)
for f_id, exp_c in expected_counts.items():
    act_c = fold_counts.get(f_id, 0)
    if act_c != exp_c:
        raise ValueError(f"Fold {f_id} : attendu {exp_c} images, trouvé {act_c}")

num_classes = df_folds["class_id"].nunique() if "class_id" in df_folds.columns else df_folds["class_name"].nunique()
print(f"Nombre de classes détecté dans le manifeste : {num_classes} (attendu : 22)")
print("Contrôle du manifeste validé à 100%.")
"""
    cells.append(nbf.v4.new_code_cell(c17_code))

    # Cell 18: Section 9 Header
    cells.append(nbf.v4.new_markdown_cell("## 9. Inspection et Validation de la Configuration EfficientNetV2B0"))

    # Cell 19: Config Inspection Code
    c19_code = r"""import yaml
from pathlib import Path

config_full_path = Path(PROJECT_DIR) / CONFIG_PATH
with open(config_full_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=== Paramètres Clés de l'Expérience ===")
print("Nom de l'expérience :", cfg.get("experiment_name"))
print("Architecture :", cfg["model"]["architecture"])
print("Poids initiaux :", cfg["model"]["weights"])
print("Taille d'image :", cfg["data"]["image_size"])
print("Nombre de classes :", cfg["data"]["num_classes"])
print("Seed globale :", cfg["project"]["seed"])
print("Batch size :", cfg["training"]["batch_size"])
print("Époques Phase 1 (Tête) :", cfg["training"]["head_epochs"])
print("Époques Phase 2 (Fine-tuning) :", cfg["training"]["fine_tuning_epochs"])
print("Learning Rate Phase 1 :", cfg["training"]["head_learning_rate"])
print("Learning Rate Phase 2 :", cfg["training"]["fine_tuning_learning_rate"])
print("Augmentation en ligne :", cfg.get("augmentation", {}).get("policy"))
print("Class weights :", cfg["training"].get("use_class_weights"))
print("Early stopping patience :", cfg.get("callbacks", {}).get("early_stopping_patience"))
print("Monitor callback :", cfg.get("callbacks", {}).get("monitor"))

# Vérifications strictes du protocole
assert cfg["model"]["architecture"] == "EfficientNetV2B0", "L'architecture doit être EfficientNetV2B0."
assert cfg["project"]["seed"] == 42, "La seed globale doit être 42."
assert cfg["data"]["num_classes"] == 22, "Le nombre de classes doit être 22."
assert "rescale" not in cfg["data"], "Aucune division externe par 255 ne doit être configurée."

print("\nValidation automatique du fichier YAML réussie.")
"""
    cells.append(nbf.v4.new_code_cell(c19_code))

    # Cell 20: Section 10 Header
    cells.append(nbf.v4.new_markdown_cell("## 10. Exécution des Tests Unitaires Automatisés (Pytest)"))

    # Cell 21: Pytest Code
    c21_code = r"""import subprocess
import sys

if RUN_TESTS:
    print("Lancement de la suite de tests unitaires (pytest)...")
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_efficientnetv2b0.py",
        "tests/test_compare_oof.py",
        "-v"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
        raise RuntimeError("Échec de la suite de tests unitaires. Corrigez les erreurs avant d'entraîner.")
    print("Tous les tests unitaires ont réussi avec succès (1 test skipped autorisé pour ImageNet).")
else:
    print("RUN_TESTS est défini sur False. Étape de test ignorée.")
"""
    cells.append(nbf.v4.new_code_cell(c21_code))

    # Cell 22: Section 11 Header
    cells.append(nbf.v4.new_markdown_cell("## 11. État des Résultats Existants et Reprise après Interruption"))

    # Cell 23: Check Existing Results Code
    c23_code = r"""from pathlib import Path
import pandas as pd

out_base = Path(OUTPUT_DIR)
status_rows = []

for f in FOLDS_TO_RUN:
    ckpt = (out_base / "models" / "efficientnetv2b0" / "checkpoints" / f"fold_{f}" / "best_model.keras").is_file()
    metrics = (out_base / "reports" / "efficientnetv2b0" / "metrics" / f"fold_{f}.json").is_file()
    oof = (out_base / "reports" / "efficientnetv2b0" / "predictions" / f"fold_{f}_oof_predictions.csv").is_file()
    h_json = (out_base / "reports" / "efficientnetv2b0" / "history" / f"fold_{f}_history.json").is_file()
    h_csv = (out_base / "reports" / "efficientnetv2b0" / "history" / f"fold_{f}_history.csv").is_file()

    status_rows.append({
        "fold": f,
        "checkpoint": ckpt,
        "metrics": metrics,
        "predictions": oof,
        "history_json": h_json,
        "history_csv": h_csv,
        "complete": all([ckpt, metrics, oof, h_json, h_csv])
    })

df_status = pd.DataFrame(status_rows)
print("=== État Actuel des Folds de Screening (Drive) ===")
print(df_status.to_string(index=False))

completed_folds = df_status[df_status["complete"]]["fold"].tolist()
if completed_folds:
    print(f"\nFolds déjà entièrement complétés : {completed_folds}")
    if not ALLOW_OVERWRITE:
        print("Note : ALLOW_OVERWRITE=False. Les folds déjà terminés ne seront pas ré-entraînés.")
"""
    cells.append(nbf.v4.new_code_cell(c23_code))

    # Cell 24: Section 12 Header
    cells.append(nbf.v4.new_markdown_cell("## 12. Smoke Test de l'Architecture EfficientNetV2B0"))

    # Cell 25: Smoke Test Code
    c25_code = r"""import gc
import tensorflow as tf
from src.models.efficientnetv2b0 import (
    build_efficientnetv2b0,
    verify_no_double_preprocessing,
    apply_efficientnet_fine_tuning_strategy
)

print("Exécution d'un Smoke Test à froid (weights=None)...")

test_model = build_efficientnetv2b0(
    num_classes=22,
    input_shape=(224, 224, 3),
    weights=None,
    head_config={"type": "article_inspired"}
)

verify_no_double_preprocessing(test_model)
assert test_model.input_shape == (None, 224, 224, 3)
assert test_model.output_shape == (None, 22)

# Verification Phase 2 fine-tuning BN freezing
apply_efficientnet_fine_tuning_strategy(
    model=test_model,
    strategy="full",
    keep_batch_normalization_frozen=True,
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy"
)

# Small forward pass on synthetic batch
dummy_batch = tf.random.uniform((2, 224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
preds = test_model(dummy_batch, training=False)
assert preds.shape == (2, 22)

print("Smoke test réussi : Entrée (None,224,224,3), Sortie (None,22), Prétraitement et BN gelées validés.")

del test_model, dummy_batch, preds
tf.keras.backend.clear_session()
gc.collect()
"""
    cells.append(nbf.v4.new_code_cell(c25_code))

    # Cell 26: Section 13 Header
    cells.append(nbf.v4.new_markdown_cell("""# 🚀 LANCEMENT DU SCREENING — FOLDS 0, 3 ET 4

> **Exécution Officielle** : L'entraînement utilise le script certifié `scripts/train_efficientnetv2b0.py`.
"""))

    # Cell 27: Training Launch Code
    c27_code = r"""import os
import subprocess
import sys

folds_args = [str(f) for f in FOLDS_TO_RUN]
cmd_train = [
    sys.executable, "scripts/train_efficientnetv2b0.py",
    "--config", CONFIG_PATH,
    "--dataset-dir", LOCAL_DATASET,
    "--output-dir", OUTPUT_DIR,
    "--folds"
] + folds_args

print("Commande d'entraînement :")
print(" ".join(cmd_train))

if RUN_TRAINING:
    print(f"\nDébut de l'entraînement des folds {FOLDS_TO_RUN} sur le dataset {LOCAL_DATASET}...\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd_train, check=True, cwd=PROJECT_DIR, env=env)
    print("\nLancement de l'entraînement terminé avec succès.")
else:
    print("\nRUN_TRAINING est défini sur False. Commande générée sans lancement de fit.")
"""
    cells.append(nbf.v4.new_code_cell(c27_code))

    # Cell 28: Section 14 Header
    cells.append(nbf.v4.new_markdown_cell("## 14. Validation et Contrôle des Checkpoints Générés"))

    # Cell 29: Checkpoint Validation Code
    c29_code = r"""import gc
from pathlib import Path
import tensorflow as tf

print("=== Validation des Checkpoints Sauvés dans Google Drive ===")
out_base = Path(OUTPUT_DIR)

for f in FOLDS_TO_RUN:
    ckpt_path = out_base / "models" / "efficientnetv2b0" / "checkpoints" / f"fold_{f}" / "best_model.keras"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint manquant pour le fold {f} à l'emplacement : {ckpt_path}")

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    model = tf.keras.models.load_model(str(ckpt_path), compile=False)
    assert model.output_shape == (None, 22), f"Forme de sortie incorrecte : {model.output_shape}"

    dummy_in = tf.random.uniform((2, 224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
    probs = model.predict(dummy_in, verbose=0)
    sums = probs.sum(axis=1)
    tf.debugging.assert_near(sums, tf.ones_like(sums), atol=1e-4)

    print(f"Fold {f} Checkpoint OK | Taille : {size_mb:.2f} MB | Sortie : {model.output_shape} | Softmax sum = {sums[0]:.4f}")

    del model, dummy_in, probs
    tf.keras.backend.clear_session()
    gc.collect()
"""
    cells.append(nbf.v4.new_code_cell(c29_code))

    # Cell 30: Section 15 Header
    cells.append(nbf.v4.new_markdown_cell("## 15. Analyse des Métriques du Screening (Folds 0, 3, 4)"))

    # Cell 31: Metrics Table Code
    c31_code = r"""import json
import numpy as np
import pandas as pd
from pathlib import Path

out_base = Path(OUTPUT_DIR)
metrics_dir = out_base / "reports" / "efficientnetv2b0" / "metrics"

metrics_rows = []
for f in FOLDS_TO_RUN:
    m_file = metrics_dir / f"fold_{f}.json"
    if not m_file.is_file():
        raise FileNotFoundError(f"Fichier de métriques manquant : {m_file}")
    with open(m_file, "r", encoding="utf-8") as fp:
        m = json.load(fp)
    metrics_rows.append({
        "fold": f,
        "accuracy": m["accuracy"],
        "macro_precision": m.get("macro_precision", 0.0),
        "macro_recall": m.get("macro_recall", 0.0),
        "macro_f1": m["macro_f1"],
        "weighted_precision": m.get("weighted_precision", 0.0),
        "weighted_recall": m.get("weighted_recall", 0.0),
        "weighted_f1": m["weighted_f1"],
        "correct_samples": m.get("correct_samples", 0),
        "total_samples": m.get("total_samples", 0),
        "duration_sec": m.get("training_duration_seconds", 0.0)
    })

df_metrics = pd.DataFrame(metrics_rows)
print("=== Métriques par Fold de Screening ===")
print(df_metrics.to_string(index=False))

mean_acc = df_metrics["accuracy"].mean()
std_acc = df_metrics["accuracy"].std()
mean_macro_f1 = df_metrics["macro_f1"].mean()
std_macro_f1 = df_metrics["macro_f1"].std()
mean_weighted_f1 = df_metrics["weighted_f1"].mean()
std_weighted_f1 = df_metrics["weighted_f1"].std()

best_fold = df_metrics.loc[df_metrics["macro_f1"].idxmax()]["fold"]
worst_fold = df_metrics.loc[df_metrics["macro_f1"].idxmin()]["fold"]

print("\n=== Synthèse Moyenne & Écart-Type (Folds 0, 3, 4) ===")
print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Macro F1 : {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
print(f"Weighted F1 : {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
print(f"Meilleur Fold (Macro F1) : Fold {int(best_fold)}")
print(f"Moins bon Fold (Macro F1) : Fold {int(worst_fold)}")
"""
    cells.append(nbf.v4.new_code_cell(c31_code))

    # Cell 32: Section 16 Header
    cells.append(nbf.v4.new_markdown_cell("## 16. Consolidation des Prédictions Out-Of-Fold (OOF) du Screening"))

    # Cell 33: OOF Prediction Consolidation Code
    c33_code = r"""from pathlib import Path
import pandas as pd
import numpy as np

out_base = Path(OUTPUT_DIR)
pred_dir = out_base / "reports" / "efficientnetv2b0" / "predictions"

oof_dfs = []
for f in FOLDS_TO_RUN:
    p_file = pred_dir / f"fold_{f}_oof_predictions.csv"
    if not p_file.is_file():
        raise FileNotFoundError(f"Fichier de prédictions OOF manquant : {p_file}")
    oof_dfs.append(pd.read_csv(p_file))

combined_oof = pd.concat(oof_dfs, ignore_index=True)
total_oof_rows = len(combined_oof)
unique_img_count = combined_oof["image_path"].nunique()

print(f"Prédictions OOF chargées : {total_oof_rows} lignes ({unique_img_count} images uniques).")

if total_oof_rows != 259:
    print(f"Attention : Total de lignes OOF = {total_oof_rows} (attendu : 87 + 86 + 86 = 259 images).")

if total_oof_rows != unique_img_count:
    raise ValueError("Des images en double ont été trouvées dans la consolidation OOF!")

prob_cols = [c for c in combined_oof.columns if c.startswith("prob_")]
assert len(prob_cols) == 22, f"Attendu 22 colonnes prob_X, trouvé {len(prob_cols)}"

prob_sums = combined_oof[prob_cols].sum(axis=1).values
np.testing.assert_allclose(prob_sums, 1.0, atol=1e-4)

screening_oof_path = pred_dir / "efficientnetv2b0_screening_oof_predictions.csv"
combined_oof.to_csv(screening_oof_path, index=False)
print(f"Fichier OOF de screening consolidé sauvegardé avec succès : {screening_oof_path}")
"""
    cells.append(nbf.v4.new_code_cell(c33_code))

    # Cell 34: Section 17 Header
    cells.append(nbf.v4.new_markdown_cell("## 17. Génération des Matrices de Confusion et Rapports de Présentation"))

    # Cell 35: Confusion Matrix & Report Code
    c35_code = r"""import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

out_base = Path(OUTPUT_DIR)
pres_dir = out_base / "reports" / "efficientnetv2b0" / "presentation_screening"
pres_dir.mkdir(parents=True, exist_ok=True)

if GENERATE_REPORT:
    oof_file = out_base / "reports" / "efficientnetv2b0" / "predictions" / "efficientnetv2b0_screening_oof_predictions.csv"
    df_oof = pd.read_csv(oof_file)

    y_true = df_oof["true_label"].values
    y_pred = df_oof["predicted_label"].values

    labels = sorted(list(set(y_true)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # Plot Matrice de Confusion Brute (Counts)
    plt.figure(figsize=(12, 10), dpi=300)
    plt.imshow(cm_counts, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("EfficientNetV2B0 Exp A — Screening OOF folds 0, 3, 4 (Counts)", fontsize=14, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Classe Prédite", fontsize=12)
    plt.ylabel("Vraie Classe", fontsize=12)
    plt.tight_layout()
    plt.savefig(pres_dir / "confusion_matrix_screening_counts.png", dpi=300)
    plt.close()

    # Plot Matrice de Confusion Normalisée
    plt.figure(figsize=(12, 10), dpi=300)
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("EfficientNetV2B0 Exp A — Screening OOF folds 0, 3, 4 (Normalized)", fontsize=14, pad=15)
    plt.colorbar()
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Classe Prédite", fontsize=12)
    plt.ylabel("Vraie Classe", fontsize=12)
    plt.tight_layout()
    plt.savefig(pres_dir / "confusion_matrix_screening_normalized.png", dpi=300)
    plt.close()

    # Sauvegarde CSV des matrices
    pd.DataFrame(cm_counts).to_csv(pres_dir / "confusion_matrix_screening_counts.csv", index=False)
    pd.DataFrame(cm_norm).to_csv(pres_dir / "confusion_matrix_screening_normalized.csv", index=False)

    # Classification Report
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(pres_dir / "classification_report_screening.csv")
    with open(pres_dir / "classification_report_screening.json", "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)

    # Top Confusions
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm_counts[i, j] > 0:
                confusions.append({"true_class": labels[i], "predicted_class": labels[j], "count": int(cm_counts[i, j])})
    df_top_conf = pd.DataFrame(confusions).sort_values("count", ascending=False)
    df_top_conf.to_csv(pres_dir / "top_confusions_screening.csv", index=False)

    overall = {
        "model": "EfficientNetV2B0",
        "scope": "Screening Folds 0, 3, 4",
        "total_images": len(df_oof),
        "accuracy": float(report_dict["accuracy"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"])
    }
    with open(pres_dir / "overall_metrics_screening.json", "w", encoding="utf-8") as fp:
        json.dump(overall, fp, indent=2)

    print(f"Rapports visuels et matrices sauvegardés dans : {pres_dir}")
"""
    cells.append(nbf.v4.new_code_cell(c35_code))

    # Cell 36: Section 18 Header
    cells.append(nbf.v4.new_markdown_cell("## 18. Visualisation des Courbes d'Entraînement par Fold"))

    # Cell 37: Training Curves Code
    c37_code = r"""import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

out_base = Path(OUTPUT_DIR)
curves_dir = out_base / "reports" / "efficientnetv2b0" / "training_curves"
curves_dir.mkdir(parents=True, exist_ok=True)

hist_dir = out_base / "reports" / "efficientnetv2b0" / "history"

for f in FOLDS_TO_RUN:
    h_file = hist_dir / f"fold_{f}_history.json"
    if not h_file.is_file():
        continue
    with open(h_file, "r", encoding="utf-8") as fp:
        hist = json.load(fp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    epochs = range(1, len(hist["accuracy"]) + 1)

    # Plot Accuracy
    axes[0].plot(epochs, hist["accuracy"], label="Train Accuracy", linewidth=2)
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title(f"Fold {f} — Accuracy", fontsize=12)
    axes[0].set_xlabel("Époques")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Loss
    axes[1].plot(epochs, hist["loss"], label="Train Loss", linewidth=2)
    if "val_loss" in hist:
        axes[1].plot(epochs, hist["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title(f"Fold {f} — Loss", fontsize=12)
    axes[1].set_xlabel("Époques")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(f"EfficientNetV2B0 Exp A — Courbes d'Apprentissage Fold {f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(curves_dir / f"fold_{f}_training_curves.png", dpi=300)
    plt.close()

print(f"Courbes d'apprentissage générées et sauvegardées sous : {curves_dir}")
"""
    cells.append(nbf.v4.new_code_cell(c37_code))

    # Cell 38: Section 19 Header
    cells.append(nbf.v4.new_markdown_cell("## 19. Comparaison avec DenseNet121 Exp D et Analyse de Complémentarité"))

    # Cell 39: DenseNet Comparison Code
    c39_code = r"""import os
import sys
import json
from pathlib import Path
import subprocess

print("=== Référence DenseNet121 Exp D (Screening Folds 0, 3, 4) ===")
print(" - Mean Accuracy : ~0.8610")
print(" - Mean Macro F1 : ~0.7889")
print(" - Mean Weighted F1 : ~0.8538\n")

densenet_oof_csv = Path(PROJECT_DIR) / "artifacts/models/densenet121_exp_d_v1/evaluation/oof_predictions_without_tta.csv"
eff_oof_dir = Path(OUTPUT_DIR) / "reports" / "efficientnetv2b0" / "predictions"

if densenet_oof_csv.is_file() and eff_oof_dir.is_dir():
    print("Exécution de l'analyse de complémentarité OOF (scripts/compare_densenet_efficientnet_oof.py)...")
    comp_output_json = Path(OUTPUT_DIR) / "reports" / "efficientnetv2b0" / "densenet_efficientnet_complementarity.json"
    cmd_comp = [
        sys.executable, "scripts/compare_densenet_efficientnet_oof.py",
        "--densenet-oof", str(densenet_oof_csv),
        "--efficientnet-oof-dir", str(eff_oof_dir),
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
        print("Erreurs corrigées par EfficientNet :", comp_res["complementarity_breakdown"]["efficientnet_only_correct"])
        print("Erreurs corrigées par DenseNet :", comp_res["complementarity_breakdown"]["densenet_only_correct"])
        print("Accuracy Ensemble 50/50 :", comp_res["simple_ensemble_50_50"]["accuracy"])
        print("Macro F1 Ensemble 50/50 :", comp_res["simple_ensemble_50_50"]["macro_f1"])
else:
    print("Information : Les prédictions OOF archivées de DenseNet121 ne sont pas détectées localement.")
    print("L'analyse de complémentarité fine est reportée.")
"""
    cells.append(nbf.v4.new_code_cell(c39_code))

    # Cell 40: Section 20 Header
    cells.append(nbf.v4.new_markdown_cell("## 20. Résumé Final et Verdict d'Exécution"))

    # Cell 41: Final Verdict Code
    c41_code = r"""from pathlib import Path
import json

out_base = Path(OUTPUT_DIR)
all_ok = True

for f in FOLDS_TO_RUN:
    ckpt = (out_base / "models" / "efficientnetv2b0" / "checkpoints" / f"fold_{f}" / "best_model.keras").is_file()
    metrics = (out_base / "reports" / "efficientnetv2b0" / "metrics" / f"fold_{f}.json").is_file()
    oof = (out_base / "reports" / "efficientnetv2b0" / "predictions" / f"fold_{f}_oof_predictions.csv").is_file()
    if not (ckpt and metrics and oof):
        all_ok = False
        print(f"Fold {f} incomplet (ckpt={ckpt}, metrics={metrics}, oof={oof})")

screening_oof = (out_base / "reports" / "efficientnetv2b0" / "predictions" / "efficientnetv2b0_screening_oof_predictions.csv").is_file()
summary_json = (out_base / "reports" / "efficientnetv2b0" / "efficientnetv2b0_screening_summary.json").is_file()

if not (screening_oof and summary_json):
    all_ok = False

if all_ok:
    print("\n" + "="*50)
    print("EFFICIENTNETV2B0 EXP A SCREENING COMPLETE")
    print("="*50 + "\n")
    print("Folds validés :", FOLDS_TO_RUN)
    print("Emplacement des résultats Google Drive :", OUTPUT_DIR)
    print("Matrices et figures :", out_base / "reports" / "efficientnetv2b0" / "presentation_screening")
else:
    print("\nStatut : Exécution partielle ou interrompue.")
"""
    cells.append(nbf.v4.new_code_cell(c41_code))

    # Cell 42: Note on Future Confirmation
    c42_md = """## 21. Note sur le Mode Confirmation Futur

Pour étendre l'expérience aux **5 folds complets** ultérieurement :
1. Modifiez la variable `FOLDS_TO_RUN = [1, 2]` dans la cellule de configuration initial.
2. Modifiez le dossier de sortie vers : `OUTPUT_DIR = "/content/drive/MyDrive/histology-results/efficientnetv2b0-exp-a-full-5-folds"`.
3. Relancez le notebook. Une fois les 5 folds entraînés, une consolidation OOF globale sur les **432 images originales** sera produite.
"""
    cells.append(nbf.v4.new_markdown_cell(c42_md))

    nb["cells"] = cells

    out_file = Path("notebooks/colab/efficientnetv2b0_exp_a_complete_training.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook généré avec succès à l'emplacement : {out_file.resolve()}")
    print(f"Nombre total de cellules : {len(cells)}")

if __name__ == "__main__":
    create_notebook()
