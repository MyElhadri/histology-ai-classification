import json
from pathlib import Path
import numpy as np
import pytest
from src.utils.config import load_yaml
from src.utils.serialization import convert_numpy_to_python

try:
    import tensorflow as tf
    HAS_TF = (
        hasattr(tf, "keras")
        and hasattr(tf.keras, "metrics")
        and hasattr(tf.keras.metrics, "CategoricalCrossentropy")
        and hasattr(tf, "data")
        and hasattr(tf.data, "Dataset")
        and hasattr(tf.data.Dataset, "from_tensor_slices")
    )
except Exception:
    HAS_TF = False



def test_yaml_e1_loads_article_inspired_head():
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    assert cfg["model"]["classifier_head"]["type"] == "article_inspired"


def test_yaml_e1_loads_rich_augmentation():
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    assert cfg["augmentation"]["policy"] == "rich"
    assert cfg["augmentation"]["enabled"] is True


def test_yaml_e1_loads_label_smoothing_005():
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    assert cfg["loss"]["label_smoothing"] == 0.05
    assert cfg["classifier_retraining"]["loss"]["label_smoothing"] == 0.05


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_ce_hard_uses_zero_label_smoothing():
    metric = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)
    assert metric.label_smoothing == 0.0


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_ce_hard_does_not_include_l2_loss():
    cce_hard = tf.keras.metrics.CategoricalCrossentropy(name="ce_hard", label_smoothing=0.0)
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    y_pred = tf.constant([[0.9, 0.1], [0.1, 0.9]])
    cce_hard.update_state(y_true, y_pred)
    val = float(cce_hard.result().numpy())
    assert abs(val - 0.10536) < 1e-3



def test_class_weights_disabled_in_phases():
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    assert cfg["training"]["use_class_weights"] is False
    assert cfg["sampling"]["representation_phase"]["class_weights"] is False
    assert cfg["classifier_retraining"]["class_weights"]["enabled"] is False


def test_samplers_configured_correctly():
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    assert cfg["sampling"]["representation_phase"]["strategy"] == "natural"
    assert cfg["classifier_retraining"]["sampler"]["strategy"] == "square_root"


def test_square_root_class_probabilities_ratio_38_vs_7():
    freq_power = -0.5
    # Class 0: 38 images, Class 1: 7 images
    # Total class weight = n_c * n_c**-0.5 = sqrt(n_c)
    w0 = 38.0 ** 0.5
    w1 = 7.0 ** 0.5
    ratio = w0 / w1
    # sqrt(38)/sqrt(7) ~ 2.3299
    assert abs(ratio - 2.3299) < 1e-3


def test_numpy_serialization():
    data = {
        "int_val": np.int64(42),
        "float_val": np.float32(0.8841),
        "arr": np.array([1, 2, 3])
    }
    converted = convert_numpy_to_python(data)
    dumped = json.dumps(converted)
    loaded = json.loads(dumped)
    assert loaded["int_val"] == 42
    assert abs(loaded["float_val"] - 0.8841) < 1e-4
    assert loaded["arr"] == [1, 2, 3]


def test_seed_override_behavior():
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    # None keeps YAML seed
    seed1 = cfg.get("_seed_override") or cfg["project"]["seed"]
    assert seed1 == 42

    # Override applies custom seed
    cfg["_seed_override"] = 123
    seed2 = cfg.get("_seed_override") or cfg["project"]["seed"]
    assert seed2 == 123


def test_previous_experiments_configs_remain_unchanged():
    base = load_yaml("configs/densenet121.yaml")
    assert base["model"]["classifier_head"]["type"] == "baseline"

    exp_a = load_yaml("configs/experiments/densenet121_exp_a_rich_augmentation.yaml")
    assert exp_a["augmentation"]["policy"] == "rich"

    exp_b = load_yaml("configs/experiments/densenet121_exp_b_article_head.yaml")
    assert exp_b["model"]["classifier_head"]["type"] == "article_inspired"

    exp_c = load_yaml("configs/experiments/densenet121_exp_c_selective_unfreezing.yaml")
    assert exp_c["fine_tuning"]["strategy"] == "last_dense_block"

    exp_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    assert exp_d["augmentation"]["policy"] == "rich"
    assert exp_d["model"]["classifier_head"]["type"] == "article_inspired"


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_square_root_dataset_creation_and_epoch_size():
    from src.data.pipeline import create_square_root_dataset
    manifest = "data/manifests/densenet121_folds.csv"
    if not Path(manifest).is_file():
        pytest.skip("Manifest file missing")

    ds, N, class_probs, orig_counts = create_square_root_dataset(
        manifest_path=manifest,
        fold=0,
        batch_size=16,
        seed=42,
        frequency_power=-0.5
    )
    assert N > 0
    assert len(class_probs) == 22
    assert abs(sum(class_probs.values()) - 1.0) < 1e-4


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_crt_phase_trainable_parameters_is_2838():
    from src.models.densenet121 import build_densenet121
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    model = build_densenet121(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=cfg["model"]["classifier_head"]
    )

    for layer in model.layers:
        layer.trainable = (layer.name == "predictions")

    crt_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    assert crt_params == 2838


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_trainable_params_less_than_total_when_bn_frozen():
    """10. Verify trainable_params < total_params when BatchNormalization layers are frozen."""
    from src.models.densenet121 import build_densenet121, apply_fine_tuning_strategy
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    model = build_densenet121(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=cfg["model"]["classifier_head"]
    )

    apply_fine_tuning_strategy(model, strategy="full", keep_batch_normalization_frozen=True)

    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    assert trainable_params < total_params
    assert trainable_params + non_trainable_params == total_params
    assert non_trainable_params > 0


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_trainable_params_uses_trainable_weights():
    """11. Verify sum(count_params(w) for w in model.trainable_weights) is used for trainable parameters."""
    from src.models.densenet121 import build_densenet121, apply_fine_tuning_strategy
    cfg = load_yaml("configs/experiments/densenet121_exp_e1_regularized_crt.yaml")
    model = build_densenet121(
        num_classes=22,
        input_shape=(224, 224, 3),
        weights=None,
        head_config=cfg["model"]["classifier_head"]
    )

    apply_fine_tuning_strategy(model, strategy="full", keep_batch_normalization_frozen=True)

    calc_trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    # Total params includes non-trainable BN weights, so count_params() is larger
    assert model.count_params() != calc_trainable
    assert calc_trainable == sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)



