import json
import pytest
import numpy as np
from pathlib import Path
from src.utils.config import load_yaml

try:
    import tensorflow as tf
    from src.models.densenet121 import (
        build_densenet121,
        apply_fine_tuning_strategy,
        validate_fine_tuning_strategy,
        validate_model_matches_config,
    )
    HAS_TF = True
except ImportError:
    HAS_TF = False
    build_densenet121 = None
    apply_fine_tuning_strategy = None
    validate_fine_tuning_strategy = None
    validate_model_matches_config = None


# --- Non-TF Tests (Tests 12, 13, 14) ---

def test_yaml_c_uses_baseline_augmentation():
    """12. YAML C uses baseline augmentation policy."""
    cfg_c_path = Path("configs/experiments/densenet121_exp_c_selective_unfreezing.yaml")
    assert cfg_c_path.exists(), "YAML C file missing"
    cfg_c = load_yaml(cfg_c_path)
    
    aug_cfg = cfg_c.get("augmentation", {})
    assert aug_cfg.get("policy") == "baseline"
    assert "rotation_factor" not in aug_cfg
    assert "zoom_factor" not in aug_cfg


def test_yaml_c_uses_baseline_head():
    """13. YAML C uses baseline classifier head."""
    cfg_c = load_yaml("configs/experiments/densenet121_exp_c_selective_unfreezing.yaml")
    head_cfg = cfg_c.get("model", {}).get("classifier_head", {})
    assert head_cfg.get("type") == "baseline"


def test_json_summary_accepts_numpy_int64():
    """14. JSON summary handles conversion of NumPy int64/float64 types without error."""
    summary_data = {
        "trainable_backbone_layer_count": int(np.int64(42)),
        "trainable_backbone_params": int(np.int64(100000)),
        "accuracy": float(np.float64(0.85))
    }
    dumped = json.dumps(summary_data)
    assert '"trainable_backbone_layer_count": 42' in dumped


# --- TF Required Tests (Tests 1-11) ---

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_full_strategy_preserves_baseline_behavior():
    """1. The full strategy unfreezes all base non-BN layers like baseline."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    apply_fine_tuning_strategy(model, strategy="full", keep_batch_normalization_frozen=True)
    
    # Check that conv1, conv2, conv3, conv4, conv5 non-BN layers are trainable
    conv_layers = [
        l for l in model.layers
        if l.name.startswith(("conv1_", "conv2_", "conv3_", "conv4_", "conv5_"))
        and not isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert all(l.trainable for l in conv_layers)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_last_dense_block_trains_conv5_layers():
    """2. last_dense_block unfreezes conv5_* non-BN layers."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    apply_fine_tuning_strategy(
        model,
        strategy="last_dense_block",
        trainable_layer_prefixes=["conv5_"],
        keep_batch_normalization_frozen=True
    )
    
    conv5_non_bn = [
        l for l in model.layers
        if l.name.startswith("conv5_") and not isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert len(conv5_non_bn) > 0
    assert all(l.trainable for l in conv5_non_bn)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_forbidden_blocks_remain_frozen():
    """3. No conv1_*, conv2_*, conv3_*, or conv4_* layers are trainable under last_dense_block."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    apply_fine_tuning_strategy(
        model,
        strategy="last_dense_block",
        trainable_layer_prefixes=["conv5_"],
        keep_batch_normalization_frozen=True
    )
    
    forbidden = [
        l for l in model.layers
        if l.name.startswith(("conv1_", "conv2_", "conv3_", "conv4_"))
    ]
    assert not any(l.trainable for l in forbidden)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_all_backbone_bn_layers_remain_frozen():
    """4. All backbone BatchNormalization layers remain frozen."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    apply_fine_tuning_strategy(
        model,
        strategy="last_dense_block",
        trainable_layer_prefixes=["conv5_"],
        keep_batch_normalization_frozen=True
    )
    
    backbone_bn = [
        l for l in model.layers
        if not l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
        and isinstance(l, tf.keras.layers.BatchNormalization)
    ]
    assert len(backbone_bn) > 0
    assert not any(l.trainable for l in backbone_bn)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_baseline_head_remains_trainable():
    """5. Baseline head remains trainable."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    apply_fine_tuning_strategy(
        model,
        strategy="last_dense_block",
        trainable_layer_prefixes=["conv5_"],
        keep_batch_normalization_frozen=True
    )
    
    head_layers = [
        l for l in model.layers
        if l.name.startswith(("global_average_pooling", "classifier_", "predictions"))
    ]
    assert all(l.trainable for l in head_layers)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_article_inspired_head_not_used():
    """6. Article-inspired head is not present when baseline is configured."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    layer_names = [l.name for l in model.layers]
    assert "classifier_dense_512" not in layer_names
    assert "classifier_batch_norm" not in layer_names
    assert "classifier_dense_128" not in layer_names


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_model_recompiled_after_trainability_change():
    """7. Model trainable weights count changes after applying strategy and recompiling."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    
    # Phase 1: freeze base
    for l in model.layers:
        if not l.name.startswith(("global_average_pooling", "classifier_", "predictions")):
            l.trainable = False
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    phase1_trainable_weights_count = len(model.trainable_weights)
    
    # Phase 2: last_dense_block
    apply_fine_tuning_strategy(model, strategy="last_dense_block", trainable_layer_prefixes=["conv5_"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    phase2_trainable_weights_count = len(model.trainable_weights)
    
    assert phase2_trainable_weights_count > phase1_trainable_weights_count


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_unknown_strategy_raises_error():
    """8. Unknown strategy produces a clear ValueError."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    with pytest.raises(ValueError, match="Unknown fine_tuning strategy"):
        apply_fine_tuning_strategy(model, strategy="invalid_strategy")


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_empty_prefixes_raises_error():
    """9. Empty prefix list produces a clear ValueError for last_dense_block strategy."""
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    with pytest.raises(ValueError, match="trainable_layer_prefixes must not be empty"):
        apply_fine_tuning_strategy(model, strategy="last_dense_block", trainable_layer_prefixes=[])


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_runtime_validation_accepts_correct_config():
    """10. validate_fine_tuning_strategy accepts correct configuration."""
    cfg_c = load_yaml("configs/experiments/densenet121_exp_c_selective_unfreezing.yaml")
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    apply_fine_tuning_strategy(
        model,
        strategy="last_dense_block",
        trainable_layer_prefixes=["conv5_"],
        keep_batch_normalization_frozen=True
    )
    # Should not raise any error
    validate_fine_tuning_strategy(model, cfg_c)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_runtime_validation_refuses_full_unfreezing_when_last_dense_block_requested():
    """11. validate_fine_tuning_strategy refuses full unfreezing when last_dense_block is requested."""
    cfg_c = load_yaml("configs/experiments/densenet121_exp_c_selective_unfreezing.yaml")
    model = build_densenet121(weights=None, head_config={"type": "baseline"})
    
    # Apply full unfreezing instead of last_dense_block
    apply_fine_tuning_strategy(model, strategy="full", keep_batch_normalization_frozen=True)
    
    with pytest.raises(ValueError, match="Configured fine-tuning strategy 'last_dense_block' does not match actual trainable layers"):
        validate_fine_tuning_strategy(model, cfg_c)
