import pytest
import numpy as np

from app.main import (
    resize_image_array,
    jpeg_bytes_to_ndarray,
    ndarray_to_jpeg_bytes,
)


# Create a sample test image
@pytest.fixture
def sample_image():
    # 100x50 RGB image with a red square
    return np.full((50, 100, 3), fill_value=(0, 0, 255), dtype=np.uint8)


def test_resize_image_array_same_dimensions(sample_image):
    result = resize_image_array(sample_image, width=100, height=50)
    assert result.shape == (50, 100, 3)


def test_resize_image_array_only_width(sample_image):
    result = resize_image_array(sample_image, width=200, height=None)
    expected_height = int(50 * (200 / 100))
    assert result.shape == (expected_height, 200, 3)


def test_resize_image_array_only_height(sample_image):
    result = resize_image_array(sample_image, width=None, height=100)
    expected_width = int(100 * (100 / 50))
    assert result.shape == (100, expected_width, 3)


def test_resize_image_array_no_resize(sample_image):
    result = resize_image_array(sample_image, width=None, height=None)
    assert result.shape == sample_image.shape
    assert np.array_equal(result, sample_image)


def test_resize_image_array_invalid_input():
    with pytest.raises(ValueError):
        resize_image_array(None, width=100, height=100)


def test_ndarray_to_jpeg_bytes_and_back(sample_image):
    jpeg_data = ndarray_to_jpeg_bytes(sample_image, quality=90)
    assert isinstance(jpeg_data, bytes)
    decoded_image = jpeg_bytes_to_ndarray(jpeg_data)
    assert isinstance(decoded_image, np.ndarray)
    assert decoded_image.shape[2] == 3  # 3 color channels


def test_jpeg_bytes_to_ndarray_invalid_data():
    with pytest.raises(ValueError):
        jpeg_bytes_to_ndarray(b"notajpeg")
