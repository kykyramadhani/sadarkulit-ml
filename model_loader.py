# File: model_loader.py (Kembali ke versi TensorFlow)

import os
import json
import tensorflow as tf # Diubah kembali
import numpy as np

MODEL_FILENAME = "my_model.tflite" # Tetap gunakan model .tflite
LABELS_FILENAME = "class_labels.json"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", MODEL_FILENAME)
LABEL_PATH = os.path.join(CURRENT_DIR, "model", LABELS_FILENAME)

interpreter = None
input_details = None
output_details = None
class_labels = None

def load_model_and_labels():
    global interpreter, class_labels, input_details, output_details
    try:
        if os.path.exists(MODEL_PATH):
            # Diubah kembali menggunakan tf.lite.Interpreter dari library tensorflow
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("Interpreter TFLite dari TensorFlow berhasil dimuat.")
        else:
            raise FileNotFoundError(f"File model TFLite tidak ditemukan di {MODEL_PATH}")

        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH, 'r') as f:
                class_labels = json.load(f)
            print("Label berhasil dimuat.")
        else:
            class_labels = {}
    except Exception as e:
        print(f"Error saat memuat model atau label: {e}")
        raise e