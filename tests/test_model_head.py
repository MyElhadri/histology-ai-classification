import pytest

try:
    import tensorflow as tf
    from src.models.densenet121 import build_densenet121
    HAS_TF = True
except ImportError:
    HAS_TF = False
    build_densenet121 = None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="TensorFlow is required for model tests")

def test_baseline_head_unchanged():
    """1. The baseline head must remain unchanged."""
    model = build_densenet121(num_classes=22, input_shape=(224, 224, 3), weights=None, dropout_rate=0.30, head_config={"type": "baseline"})
    # Baseline has GlobalAveragePooling2D -> Dropout -> Dense(22, softmax)
    layers = model.layers
    # The last 3 layers should be exactly what we expect
    assert isinstance(layers[-3], tf.keras.layers.GlobalAveragePooling2D)
    assert isinstance(layers[-2], tf.keras.layers.Dropout)
    assert layers[-2].rate == 0.30
    assert isinstance(layers[-1], tf.keras.layers.Dense)
    assert layers[-1].units == 22
    assert layers[-1].activation.__name__ == "softmax"


def test_article_inspired_head():
    """Tests 2 to 10: Verify the article_inspired head properties."""
    head_config = {
        "type": "article_inspired",
        "pooling": "global_average",
        "dense_1_units": 512,
        "dense_1_activation": "elu",
        "batch_normalization": True,
        "dropout_rate": 0.30,
        "dense_2_units": 128,
        "dense_2_activation": "elu",
        "l2_strength": 0.01,
        "output_activation": "softmax"
    }
    
    model = build_densenet121(num_classes=22, input_shape=(224, 224, 3), weights=None, head_config=head_config)
    layers = model.layers
    
    # 9. GlobalAveragePooling is used
    assert any(isinstance(l, tf.keras.layers.GlobalAveragePooling2D) for l in layers)
    # 10. Flatten is not used
    assert not any(isinstance(l, tf.keras.layers.Flatten) for l in layers)
    
    # We expect the last layers to be: GAP -> Dense(512) -> BatchNorm -> Dropout -> Dense(128) -> Dense(22)
    # Let's extract the custom head layers (those with specific names or just from the end)
    head_layers = layers[-6:]
    
    assert isinstance(head_layers[0], tf.keras.layers.GlobalAveragePooling2D)
    
    # 2. Contains Dense 512, 3. Activation ELU
    assert isinstance(head_layers[1], tf.keras.layers.Dense)
    assert head_layers[1].units == 512
    assert head_layers[1].activation.__name__ == "elu"
    
    # 4. Contains BatchNormalization
    assert isinstance(head_layers[2], tf.keras.layers.BatchNormalization)
    
    # 5. Contains Dropout 0.30
    assert isinstance(head_layers[3], tf.keras.layers.Dropout)
    assert head_layers[3].rate == 0.30
    
    # 6. Contains Dense 128, 7. L2=0.01
    assert isinstance(head_layers[4], tf.keras.layers.Dense)
    assert head_layers[4].units == 128
    assert head_layers[4].activation.__name__ == "elu"
    reg = head_layers[4].kernel_regularizer
    assert reg is not None
    assert isinstance(reg, tf.keras.regularizers.L2)
    assert abs(reg.l2 - 0.01) < 1e-6
    
    # 8. Output has 22 softmax neurons
    assert isinstance(head_layers[5], tf.keras.layers.Dense)
    assert head_layers[5].units == 22
    assert head_layers[5].activation.__name__ == "softmax"


def test_invalid_config_raises_error():
    """11. Invalid configuration causes a clear error."""
    # Unknown type
    with pytest.raises(ValueError, match="Unknown classifier_head type"):
        build_densenet121(weights=None, head_config={"type": "unknown_head"})
        
    # Negative units
    with pytest.raises(ValueError, match="strictly positive required"):
        head_config = {
            "type": "article_inspired",
            "dense_1_units": -512,
            "dense_2_units": 128
        }
        build_densenet121(weights=None, head_config=head_config)
        
    # Invalid dropout
    with pytest.raises(ValueError, match="between 0 and 1 required"):
        head_config = {
            "type": "article_inspired",
            "dense_1_units": 512,
            "dense_2_units": 128,
            "dropout_rate": 1.5
        }
        build_densenet121(weights=None, head_config=head_config)


def test_forward_pass_dummy_tensor():
    """13. The model can perform a forward pass on a small dummy tensor."""
    head_config = {
        "type": "article_inspired",
        "dense_1_units": 512,
        "dense_2_units": 128
    }
    model = build_densenet121(num_classes=22, input_shape=(224, 224, 3), weights=None, head_config=head_config)
    
    # Create a batch of 2 dummy images
    dummy_input = tf.random.normal((2, 224, 224, 3))
    output = model(dummy_input, training=False)
    
    assert output.shape == (2, 224) # wait, it's 22 output classes
    assert output.shape == (2, 22)
    
    # Values should sum to roughly 1 (softmax)
    sums = tf.reduce_sum(output, axis=-1)
    tf.debugging.assert_near(sums, tf.ones_like(sums))
