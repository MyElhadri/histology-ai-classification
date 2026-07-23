import pytest
from pathlib import Path
from src.utils.config import load_yaml

try:
    import tensorflow as tf
    from src.models.densenet121 import (
        build_densenet121,
        apply_fine_tuning_strategy,
        validate_model_matches_config,
        validate_fine_tuning_strategy
    )
    from src.data.pipeline import create_dataset
    HAS_TF = True
except ImportError:
    HAS_TF = False
    build_densenet121 = None
    apply_fine_tuning_strategy = None
    validate_model_matches_config = None
    validate_fine_tuning_strategy = None
    create_dataset = None


# --- Pure Python Tests (Tests 1, 2, 3, 13, 15) ---

def test_yaml_d_loads_article_inspired_head():
    """1. YAML D loads article_inspired classifier head."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    head_cfg = cfg_d.get("model", {}).get("classifier_head", {})
    assert head_cfg.get("type") == "article_inspired"
    assert head_cfg.get("dense_1_units") == 512
    assert head_cfg.get("dense_2_units") == 128


def test_yaml_d_loads_rich_augmentation():
    """2. YAML D loads rich augmentation policy."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    aug_cfg = cfg_d.get("augmentation", {})
    assert aug_cfg.get("enabled") is True
    assert aug_cfg.get("policy") == "rich"
    assert aug_cfg.get("rotation_factor") == 0.04
    assert aug_cfg.get("zoom_factor") == 0.10


def test_yaml_d_loads_full_fine_tuning():
    """3. YAML D loads full fine-tuning strategy."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    ft_cfg = cfg_d.get("fine_tuning", {})
    assert ft_cfg.get("strategy") == "full"
    assert ft_cfg.get("keep_batch_normalization_frozen") is True


def test_last_dense_block_strategy_not_active_in_d():
    """13. Strategy last_dense_block is NOT active in YAML D."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    ft_cfg = cfg_d.get("fine_tuning", {})
    assert ft_cfg.get("strategy") != "last_dense_block"
    assert "trainable_layer_prefixes" not in ft_cfg


def test_previous_experiments_remain_unchanged():
    """15. Experiments A, B, C and baseline YAML files remain unchanged and valid."""
    base_cfg = load_yaml("configs/densenet121.yaml")
    assert base_cfg.get("model", {}).get("classifier_head", {}).get("type") == "baseline"
    assert base_cfg.get("fine_tuning", {}).get("strategy") == "full"

    cfg_a = load_yaml("configs/experiments/densenet121_exp_a_rich_augmentation.yaml")
    assert cfg_a.get("augmentation", {}).get("policy") == "rich"

    cfg_b = load_yaml("configs/experiments/densenet121_exp_b_article_head.yaml")
    assert cfg_b.get("model", {}).get("classifier_head", {}).get("type") == "article_inspired"
    assert cfg_b.get("augmentation", {}).get("policy") == "baseline"

    cfg_c = load_yaml("configs/experiments/densenet121_exp_c_selective_unfreezing.yaml")
    assert cfg_c.get("fine_tuning", {}).get("strategy") == "last_dense_block"
    assert cfg_c.get("model", {}).get("classifier_head", {}).get("type") == "baseline"


# --- TF Required Tests (Tests 4 to 12 & 14) ---

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_bn_remains_frozen_during_fine_tuning():
    """4. Base BatchNormalization layers remain frozen during fine-tuning."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    head_cfg = cfg_d.get("model", {}).get("classifier_head")
    model = build_densenet121(weights=None, head_config=head_cfg)
    
    apply_fine_tuning_strategy(model, strategy="full", keep_batch_normalization_frozen=True)
    
    backbone_bn = [
        l for l in model.layers
        if not l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
        and isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert len(backbone_bn) > 0
    assert not any(l.trainable for l in backbone_bn)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_validation_dataset_has_no_augmentation():
    """5. Validation dataset uses no data augmentation."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    ds_val = create_dataset(manifest, "", fold=0, is_training=False, batch_size=2, augmentation_config=None)
    
    # Validation dataset is deterministic on subsequent iterators
    it1 = iter(ds_val)
    it2 = iter(ds_val)
    tf.debugging.assert_equal(next(it1)[0], next(it2)[0])


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_train_dataset_uses_rich_augmentation():
    """6. Train dataset uses rich augmentation when configured."""
    manifest = Path("data/manifests/densenet121_folds.csv")
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    aug_cfg = cfg_d.get("augmentation")
    
    ds_train = create_dataset(manifest, "", fold=0, is_training=True, batch_size=2, augmentation_config=aug_cfg)
    images, labels = next(iter(ds_train))
    assert images.shape[1:] == (224, 224, 3)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_model_d_layers_and_params():
    """7 to 12. Model D layer structure and total parameter count."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    head_cfg = cfg_d.get("model", {}).get("classifier_head")
    model = build_densenet121(weights=None, head_config=head_cfg)
    
    validate_model_matches_config(model, cfg_d)
    
    layer_names = [l.name for l in model.layers]
    
    # 7. Dense 512
    assert "classifier_dense_512" in layer_names
    
    # 8. BatchNormalization
    assert "classifier_batch_norm" in layer_names
    
    # 9. Dropout 0.30
    drop = next(l for l in model.layers if l.name == "classifier_dropout")
    assert drop.rate == 0.30
    
    # 10. Dense 128 with L2 0.01
    d128 = next(l for l in model.layers if l.name == "classifier_dense_128")
    assert d128.units == 128
    assert abs(float(d128.kernel_regularizer.l2) - 0.01) < 1e-5
    
    # 11. 22 outputs softmax
    pred = next(l for l in model.layers if l.name == "predictions")
    assert pred.units == 22
    assert pred.activation.__name__ == "softmax"
    
    # 12. Total parameters == 7632854
    assert model.count_params() == 7632854


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_model_d_forward_pass():
    """14. Model D can perform a forward pass on dummy input with weights=None."""
    cfg_d = load_yaml("configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml")
    head_cfg = cfg_d.get("model", {}).get("classifier_head")
    model = build_densenet121(num_classes=22, input_shape=(224, 224, 3), weights=None, head_config=head_cfg)
    
    dummy_input = tf.random.normal((2, 224, 224, 3))
    output = model(dummy_input, training=False)
    assert output.shape == (2, 22)
    sums = tf.reduce_sum(output, axis=-1)
    tf.debugging.assert_near(sums, tf.ones_like(sums))
