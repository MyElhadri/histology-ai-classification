import json
import pytest
from pathlib import Path
from src.utils.config import load_yaml

try:
    import tensorflow as tf
    from src.models.densenet121 import build_densenet121, validate_model_matches_config
    HAS_TF = True
except ImportError:
    HAS_TF = False
    build_densenet121 = None
    validate_model_matches_config = None


# --- Non-TF tests (Tests 12 & 13) ---

def test_yaml_b_uses_baseline_augmentation():
    """13. YAML B uses baseline augmentation policy."""
    config_b_path = Path("configs/experiments/densenet121_exp_b_article_head.yaml")
    assert config_b_path.exists(), "YAML B file missing"
    config_b = load_yaml(config_b_path)
    
    aug_cfg = config_b.get("augmentation", {})
    assert aug_cfg.get("policy") == "baseline", f"Expected policy baseline, got {aug_cfg.get('policy')}"
    # Verify rich augmentation parameters are not present
    assert "rotation_factor" not in aug_cfg
    assert "zoom_factor" not in aug_cfg


def test_notebook_does_not_overwrite_config_path():
    """12. The notebook does not overwrite CONFIG_PATH in later cells."""
    nb_path = Path("notebooks/colab/02_train_densenet121.ipynb")
    assert nb_path.exists(), "Colab notebook missing"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    config_path_assignments = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for line in cell["source"]:
                line_str = line.strip()
                if (line_str.startswith("CONFIG_PATH =") or line_str.startswith("CONFIG_PATH=")) and not line_str.startswith("#"):
                    config_path_assignments.append(line_str)
                    
    # Only load-yaml cell should assign CONFIG_PATH
    assert len(config_path_assignments) == 1, f"Multiple CONFIG_PATH assignments found: {config_path_assignments}"
    assert "configs/" in config_path_assignments[0] and ".yaml" in config_path_assignments[0]


# --- TF required tests (Tests 1-11) ---

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for model head tests")
def test_yaml_baseline_constructs_baseline_head():
    """1. YAML baseline constructs baseline head."""
    cfg_base = load_yaml("configs/densenet121.yaml")
    head_cfg = cfg_base.get("model", {}).get("classifier_head") or cfg_base.get("classifier_head", {"type": "baseline"})
    model = build_densenet121(weights=None, head_config=head_cfg)
    
    layer_names = [l.name for l in model.layers]
    assert "predictions" in layer_names
    assert "classifier_dense_512" not in layer_names
    assert model.count_params() == 7060054


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for model head tests")
def test_yaml_b_constructs_article_inspired_head():
    """2 to 8. YAML B constructs article_inspired head with expected layers."""
    cfg_b = load_yaml("configs/experiments/densenet121_exp_b_article_head.yaml")
    head_cfg = cfg_b.get("model", {}).get("classifier_head")
    model = build_densenet121(weights=None, head_config=head_cfg)
    
    layer_names = [l.name for l in model.layers]
    
    # 2. Contains Dense 512
    assert "classifier_dense_512" in layer_names
    d512 = next(l for l in model.layers if l.name == "classifier_dense_512")
    assert d512.units == 512
    
    # 3. ELU activation present
    assert "classifier_elu_512" in layer_names
    
    # 4. BatchNormalization present
    assert "classifier_batch_norm" in layer_names
    
    # 5. Dropout 0.30 present
    assert "classifier_dropout" in layer_names
    drop = next(l for l in model.layers if l.name == "classifier_dropout")
    assert drop.rate == 0.30
    
    # 6. Dense 128 with L2 0.01
    assert "classifier_dense_128" in layer_names
    d128 = next(l for l in model.layers if l.name == "classifier_dense_128")
    assert d128.units == 128
    assert d128.kernel_regularizer is not None
    assert abs(float(d128.kernel_regularizer.l2) - 0.01) < 1e-5
    
    # 7. 22 outputs softmax
    assert "predictions" in layer_names
    pred = next(l for l in model.layers if l.name == "predictions")
    assert pred.units == 22
    assert pred.activation.__name__ == "softmax"
    
    # 8. Does not use Flatten
    assert not any(isinstance(l, tf.keras.layers.Flatten) for l in model.layers)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for model head tests")
def test_param_count_b_greater_than_baseline():
    """9. Number of parameters in B is greater than baseline (7,632,854 vs 7,060,054)."""
    cfg_base = load_yaml("configs/densenet121.yaml")
    head_base = cfg_base.get("model", {}).get("classifier_head")
    model_base = build_densenet121(weights=None, head_config=head_base)
    
    cfg_b = load_yaml("configs/experiments/densenet121_exp_b_article_head.yaml")
    head_b = cfg_b.get("model", {}).get("classifier_head")
    model_b = build_densenet121(weights=None, head_config=head_b)
    
    assert model_b.count_params() > model_base.count_params()
    assert model_b.count_params() == 7632854
    assert model_base.count_params() == 7060054


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for model head tests")
def test_validate_model_matches_config_success():
    """10. validate_model_matches_config accepts correct models."""
    cfg_b = load_yaml("configs/experiments/densenet121_exp_b_article_head.yaml")
    head_b = cfg_b.get("model", {}).get("classifier_head")
    model_b = build_densenet121(weights=None, head_config=head_b)
    
    # Should not raise any error
    validate_model_matches_config(model_b, cfg_b)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for model head tests")
def test_validate_model_matches_config_rejection():
    """11. validate_model_matches_config refuses a baseline model when article_inspired head is requested."""
    cfg_base = load_yaml("configs/densenet121.yaml")
    head_base = cfg_base.get("model", {}).get("classifier_head")
    model_base = build_densenet121(weights=None, head_config=head_base)
    
    cfg_b = load_yaml("configs/experiments/densenet121_exp_b_article_head.yaml")
    
    with pytest.raises(ValueError, match="Configured head 'article_inspired' does not match constructed model"):
        validate_model_matches_config(model_base, cfg_b)
