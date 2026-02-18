from pathlib import Path
import numpy as np

from src.data.read_img import read_dicom_image


def test_read_dicom_image_returns_rgb_array_and_pil():
    dicom_dir = Path("data/raw/DICOM")
    dcm_files = list(dicom_dir.glob("*.dcm"))

    assert len(dcm_files) > 0, "No se encontraron archivos .dcm en data/raw/DICOM"

    rgb_array, pil_image = read_dicom_image(str(dcm_files[0]))

    # Validar array
    assert isinstance(rgb_array, np.ndarray)
    assert rgb_array.ndim == 3
    assert rgb_array.shape[2] == 3  # RGB
    assert rgb_array.dtype == np.uint8

    # Validar PIL
    assert pil_image is not None
    assert hasattr(pil_image, "size")
