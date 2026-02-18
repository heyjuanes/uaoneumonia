import numpy as np
import tensorflow as tf

from src.visualizations.grad_cam import generate_gradcam


def _build_tiny_model(layer_name: str = "conv10_thisone") -> tf.keras.Model:
    """
    Construye un modelo CNN pequeño con una capa conv nombrada como espera Grad-CAM.
    Entrada: (512,512,1)
    Salida: 3 clases (softmax)
    """
    inputs = tf.keras.Input(shape=(512, 512, 1), name="input")
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same", name=layer_name)(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="preds")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def test_generate_gradcam_returns_uint8_rgb_image():
    layer_name = "conv10_thisone"
    model = _build_tiny_model(layer_name=layer_name)

    # Imagen batch normalizada (1,512,512,1)
    image_batch = np.random.rand(1, 512, 512, 1).astype(np.float32)

    # Imagen original RGB uint8 (512,512,3) para overlay
    original_rgb = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

    # class_index: forzamos una clase válida (0,1,2)
    heatmap_rgb = generate_gradcam(
        model=model,
        image_batch=image_batch,
        original_rgb=original_rgb,
        layer_name=layer_name,
        class_index=0,
    )

    assert isinstance(heatmap_rgb, np.ndarray)
    assert heatmap_rgb.shape == (512, 512, 3)
    assert heatmap_rgb.dtype == np.uint8
