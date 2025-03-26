from typing import Optional

import cv2
import make87
import numpy as np
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG


def resize_image_array(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """
    Resize an in-memory image (as a NumPy array) to the specified width and height. If both width and height are None,
    the original image is returned. If only one dimension is provided, the other dimension is calculated to maintain the
    aspect ratio.

    Args:
        image: Input image in memory.
        width: Target width. If set to None, the aspect ratio is maintained.
        height: Target height. If set to None, the aspect ratio is maintained.

    Returns:
        np.ndarray: The resized image.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid numpy ndarray image.")

    if width is None and height is None:
        return image  # No resizing needed

    if width is None:
        width = int(image.shape[1] * (height / image.shape[0]))
    elif height is None:
        height = int(image.shape[0] * (width / image.shape[1]))

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def jpeg_bytes_to_ndarray(jpeg_bytes: bytes) -> np.ndarray:
    """
    Convert JPEG bytes into a NumPy ndarray image.

    Args:
        jpeg_bytes: JPEG image data.

    Returns:
        np.ndarray: Decoded image as a NumPy array.
    """
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Use IMREAD_UNCHANGED for alpha channel
    if image is None:
        raise ValueError("Could not decode image from bytes")
    return image


def ndarray_to_jpeg_bytes(image: np.ndarray, quality: int = 95) -> bytes:
    """
    Convert a NumPy ndarray image to JPEG bytes.

    Args:
        image: Input image in BGR format (as used by OpenCV).
        quality: JPEG quality (0 to 100).

    Returns:
        bytes: JPEG-encoded image data.
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise ValueError("Could not encode image to JPEG")
    return encoded_image.tobytes()


def main():
    make87.initialize()
    input_images = make87.get_subscriber(name="INPUT_IMAGE", message_type=ImageJPEG)
    output_images = make87.get_publisher(name="OUTPUT_IMAGE", message_type=ImageJPEG)

    def callback(message: ImageJPEG):
        # Decode the JPEG bytes to a NumPy array
        image_array = jpeg_bytes_to_ndarray(jpeg_bytes=message.data)

        # Resize the image
        resized_image = resize_image_array(image=image_array, width=1920, height=480)

        # Convert the resized image back to JPEG bytes
        jpeg_bytes = ndarray_to_jpeg_bytes(image=resized_image, quality=95)

        # Create a new ImageJPEG message
        output_message = ImageJPEG()
        output_message.data = jpeg_bytes
        output_message.header.timestamp = message.header.timestamp  # Keep the original timestamp

        # Publish the resized image
        output_images.publish(message=output_message)

    input_images.subscribe(callback)
    make87.loop()


if __name__ == "__main__":
    main()
