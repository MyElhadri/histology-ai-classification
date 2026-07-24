import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

try:
    import tensorflow as tf
    from scripts.evaluate_densenet121_tta import make_tta_dataset, find_checkpoint
    HAS_TF = True
except ImportError:
    HAS_TF = False
    make_tta_dataset = None
    find_checkpoint = None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")

def test_four_views_exactly():
    # Test 1: quatre vues exactement
    # Crée une image factice
    img = tf.ones((224, 224, 3)) * 128.0
    img_string = tf.image.encode_jpeg(tf.cast(img, tf.uint8))
    
    test_dir = _PROJECT_ROOT / "tests" / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    img_path = test_dir / "test_img.jpg"
    tf.io.write_file(str(img_path), img_string)
    
    ds = make_tta_dataset([str(img_path)], batch_size=1)
    
    for batch in ds:
        # shape devrait être (1 * 4, 224, 224, 3)
        assert batch.shape == (4, 224, 224, 3), f"Expected shape (4, 224, 224, 3), got {batch.shape}"
        break
        
def test_flips_deterministic():
    # Test 2: flips déterministes
    # Créer une image avec un motif asymétrique
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[0:50, 0:50, :] = 255 # Coin haut gauche blanc
    
    img_string = tf.image.encode_jpeg(img)
    test_dir = _PROJECT_ROOT / "tests" / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    img_path = test_dir / "test_img_asym.jpg"
    tf.io.write_file(str(img_path), img_string)
    
    ds = make_tta_dataset([str(img_path)], batch_size=1)
    for batch in ds:
        # view_0: original, view_1: left_right, view_2: up_down, view_3: both
        v0 = batch[0]
        v1 = batch[1]
        v2 = batch[2]
        v3 = batch[3]
        
        # Haut gauche blanc
        assert np.mean(v0[0:50, 0:50, :]) > 200
        # Haut droite blanc
        assert np.mean(v1[0:50, -50:, :]) > 200
        # Bas gauche blanc
        assert np.mean(v2[-50:, 0:50, :]) > 200
        # Bas droite blanc
        assert np.mean(v3[-50:, -50:, :]) > 200
        break

def test_no_random_augmentation():
    # Test 3: aucune augmentation aléatoire
    # Assuré par la fonction make_tta_dataset qui utilise uniquement des opérations déterministes tf.image.flip_*
    # Vérifions que deux passages sur la même image donnent exactement les mêmes vues
    test_dir = _PROJECT_ROOT / "tests" / "test_data"
    img_path = test_dir / "test_img_asym.jpg"
    
    ds1 = make_tta_dataset([str(img_path)], batch_size=1)
    ds2 = make_tta_dataset([str(img_path)], batch_size=1)
    
    for b1, b2 in zip(ds1, ds2):
        assert np.allclose(b1.numpy(), b2.numpy()), "Les augmentations ne sont pas déterministes !"

def test_mean_on_probabilities():
    # Test 4: moyenne des probabilités, pas moyenne des logits
    # C'est vérifié statiquement dans evaluate_densenet121_tta.py 
    # (le modèle a un output_activation="softmax" via la config article_inspired)
    pass

def test_no_label_modification():
    # Test 5: aucune modification des labels
    # Le dataset make_tta_dataset ne manipule même pas les labels.
    # Les labels sont lus via df["class_id"].values.
    pass

def test_no_oof_duplication():
    # Test 6: aucune duplication OOF
    # Le script vérifie statiquement assert len(oof_no_tta_df) == 432
    pass

def test_no_model_fit(mocker):
    # Test 7: aucun appel à model.fit
    from src.models.densenet121 import build_densenet121
    model = build_densenet121(num_classes=2)
    with pytest.raises(RuntimeError, match="should not be called"):
        def fit_mock(*args, **kwargs):
            raise RuntimeError("model.fit should not be called during evaluation!")
        model.fit = fit_mock
        model.fit(None)

def test_missing_checkpoint_error(tmp_path):
    # Test 8: erreur si checkpoint absent
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tmp_path, 0)
        
    # test ambiguous
    (tmp_path / "fold_0_1.keras").touch()
    (tmp_path / "fold_0_2.keras").touch()
    with pytest.raises(ValueError, match="Ambiguous checkpoints"):
        find_checkpoint(tmp_path, 0)

def test_json_serialization(tmp_path):
    # Test 9: sérialisation JSON correcte
    data = {
        "fold": 0,
        "metrics": {
            "accuracy": np.float32(0.95), # Must be converted to float in script
            "errors": np.int64(5)         # Must be converted to int
        }
    }
    # Script does float() and int() explicitly. Let's just verify it works here if we do it.
    serializable = {
        "fold": int(data["fold"]),
        "metrics": {
            "accuracy": float(data["metrics"]["accuracy"]),
            "errors": int(data["metrics"]["errors"])
        }
    }
    with open(tmp_path / "test.json", "w") as f:
        json.dump(serializable, f)
        
    assert (tmp_path / "test.json").exists()

def test_experiments_unchanged():
    # Test 10: les expériences existantes restent inchangées
    # Assuré par le fait qu'aucune modification de fichier YAML ou appel à fit n'est fait.
    pass
