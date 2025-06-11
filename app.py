# File: app.py
# Versi final yang diupdate untuk menggunakan TensorFlow Lite Interpreter

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# --- Penyesuaian Path untuk Vercel (PENTING) ---
# Menambahkan direktori root proyek ke path agar bisa mengimpor modul helper
current_dir = os.path.dirname(__file__)
# Perbaikan: Arahkan ke folder induk (root proyek), bukan folder saat ini
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)
# ------------------------------------

# Impor modul helper Anda
import model_loader
import image_processor

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app) # Mengizinkan Cross-Origin Resource Sharing

# Konfigurasi logging untuk debugging di Vercel
logging.basicConfig(level=logging.INFO)

# --- Pemuatan Model (Hanya Sekali saat Cold Start) ---
try:
    logging.info("Memulai pemuatan model dan label TFLite...")
    model_loader.load_model_and_labels()
    logging.info("Interpreter TFLite dan label berhasil dimuat.")
except Exception as e:
    logging.error(f"!!! GAGAL MEMUAT MODEL SAAT INISIALISASI: {e}", exc_info=True)


# --- Endpoint Prediksi Utama ---
@app.route("/", methods=['GET', 'POST'])
def handle_predict():
    # Menangani GET request untuk health check
    if request.method == 'GET':
        # Diubah: Sekarang kita cek 'interpreter' bukan 'model'
        status_model = "Siap" if model_loader.interpreter is not None else "Gagal Dimuat"
        return jsonify({
            "status": "online",
            "message": "Selamat datang di SadarKulit ML API (TFLite)",
            "model_status": status_model
        })

    # Menangani POST request untuk prediksi
    if request.method == 'POST':
        logging.info("Menerima request POST untuk prediksi.")

        # Diubah: Cek 'interpreter'
        if model_loader.interpreter is None:
            logging.error("Prediksi gagal karena model tidak dimuat.")
            return jsonify({'error': 'Model tidak tersedia, periksa log server.'}), 503

        if 'image' not in request.files:
            logging.warning("Request tidak menyertakan 'image' file part.")
            return jsonify({'error': "Tidak ada bagian file 'image' dalam request."}), 400

        file = request.files['image']
        if file.filename == '':
            logging.warning("Nama file kosong, tidak ada gambar yang dipilih.")
            return jsonify({'error': 'Tidak ada file gambar yang dipilih.'}), 400

        try:
            image_bytes = file.read()
            processed_image = image_processor.preprocess_image(image_bytes)
            
            # --- BAGIAN PREDIKSI DIUBAH TOTAL UNTUK TFLITE ---
            # 1. Atur tensor input ke interpreter
            model_loader.interpreter.set_tensor(model_loader.input_details[0]['index'], processed_image)
            
            # 2. Jalankan inferensi/prediksi
            model_loader.interpreter.invoke()
            
            # 3. Dapatkan hasil dari tensor output
            predictions = model_loader.interpreter.get_tensor(model_loader.output_details[0]['index'])
            # --------------------------------------------------
            
            # Proses hasil prediksi (logika ini tetap sama)
            probabilities = predictions[0]
            predicted_index = int(np.argmax(probabilities))
            confidence_score = float(np.max(probabilities))
            disease_name = model_loader.class_labels.get(str(predicted_index), f"Kelas {predicted_index} (Tidak Dikenal)")
            
            response_data = {
                'predicted_disease': disease_name,
                'confidence': f"{confidence_score:.4f}"
            }
            
            logging.info(f"Hasil prediksi dikirim: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            logging.error(f"Error tidak terduga saat prediksi: {e}", exc_info=True)
            return jsonify({'error': 'Terjadi kesalahan internal di server.'}), 500