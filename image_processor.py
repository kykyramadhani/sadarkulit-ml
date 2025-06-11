# File: image_processor.py (Kembali ke versi TensorFlow)

from PIL import Image
import numpy as np
import io
import tensorflow as tf # Impor lagi TensorFlow
import logging

logger = logging.getLogger(__name__)

TARGET_IMAGE_WIDTH = 256
TARGET_IMAGE_HEIGHT = 256

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT))
        image_array = np.array(image, dtype=np.float32)
        image_array_expanded = np.expand_dims(image_array, axis=0)

        # Diubah kembali: Gunakan lagi fungsi bawaan TensorFlow yang lebih akurat
        processed_image_array = tf.keras.applications.efficientnet.preprocess_input(image_array_expanded)
        
        return processed_image_array
    except Exception as e:
        raise ValueError(f"Gagal memproses gambar input: {e}")