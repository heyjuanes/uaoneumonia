import numpy as np

from src.features.preprocess_img import preprocess_image


def test_preprocess_image_output_shape_dtype_range():
    # Imagen falsa RGB 512x512 (uint8)
    img = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

    batch = preprocess_image(img)

    # Debe ser batch (1, 512, 512, 1)
    assert batch.shape == (1, 512, 512, 1)

    # Debe ser float (normalizado)
    assert batch.dtype in (np.float32, np.float64)

    # Valores normalizados [0, 1]
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0
