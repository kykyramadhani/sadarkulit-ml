# File: model_loader.py
# Versi terbaru yang diupdate untuk memuat model .tflite

import os
import json
import tensorflow as tf
import numpy as np

# --- GANTI DENGAN NAMA FILE MODEL TFLITE ANDA ---
MODEL_FILENAME = "my_model.tflite"  # Pastikan nama ini sesuai dengan file Anda
# ----------------------------------------------
LABELS_FILENAME = "class_labels.json"

# --- Path dinamis (tidak perlu diubah) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", MODEL_FILENAME)
LABEL_PATH = os.path.join(CURRENT_DIR, "model", LABELS_FILENAME)

# --- Variabel global diubah untuk TFLite ---
interpreter = None      # Kita sekarang menggunakan 'interpreter', bukan 'model'
input_details = None    # Untuk menyimpan detail input model
output_details = None   # Untuk menyimpan detail output model
class_labels = None
# -----------------------------------------

def load_model_and_labels():
    global interpreter, class_labels, input_details, output_details
    
    print(f"Mencoba memuat model TFLite dari: {MODEL_PATH}")
    try:
        # --- Bagian ini diubah total untuk TFLite ---
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()  # Langkah wajib untuk alokasi memori
            
            # Dapatkan detail input dan output dari model untuk digunakan nanti
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print("Interpreter TFLite berhasil dimuat dan dialokasikan.")
        else:
            msg = f"ERROR KRITIS: File model '{MODEL_FILENAME}' tidak ditemukan di '{MODEL_PATH}'."
            print(msg)
            raise FileNotFoundError(msg)

        # Bagian memuat label tetap sama
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH, 'r', encoding='utf-8') as f:
                class_labels = json.load(f)
            print(f"Label kelas berhasil dimuat dari: {LABEL_PATH}")
        else:
            print(f"PERINGATAN: File label kelas '{LABELS_FILENAME}' tidak ditemukan di '{LABEL_PATH}'.")
            class_labels = {} 

    except Exception as e:
        print(f"Terjadi kesalahan fatal saat memuat model atau label: {e}")
        raise e