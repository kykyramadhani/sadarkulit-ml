# File: app.py
# Versi final yang siap untuk di-deploy ke Vercel

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# --- Penyesuaian Path untuk Vercel ---
# Menambahkan direktori root proyek ke path agar bisa mengimpor modul helper
# Ini penting agar 'import model_loader' berfungsi saat di-deploy
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)
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
# Memuat model dan label saat serverless function pertama kali dijalankan.
# Ini jauh lebih efisien.
try:
    logging.info("Memulai pemuatan model dan label...")
    model_loader.load_model_and_labels()
    logging.info("Model dan label berhasil dimuat.")
except Exception as e:
    logging.error(f"!!! GAGAL MEMUAT MODEL SAAT INISIALISASI: {e}", exc_info=True)
# ----------------------------------------------------


# --- Endpoint Prediksi Utama ---
# Vercel akan mengarahkan semua request ke file ini
@app.route("/", methods=['GET', 'POST'])
def handle_predict():
    # Menangani GET request (misalnya untuk health check)
    if request.method == 'GET':
        status_model = "Siap" if model_loader.model is not None else "Gagal Dimuat"
        return jsonify({
            "status": "online",
            "message": "Selamat datang di SadarKulit ML API",
            "model_status": status_model
        })

    # Menangani POST request untuk prediksi
    if request.method == 'POST':
        logging.info("Menerima request POST untuk prediksi.")

        if model_loader.model is None:
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
            
            # 1. Pra-pemrosesan Gambar
            processed_image = image_processor.preprocess_image(image_bytes)
            
            # 2. Lakukan Prediksi
            predictions = model_loader.model.predict(processed_image)
            
            # 3. Proses Hasil Prediksi
            probabilities = predictions[0]
            predicted_index = int(np.argmax(probabilities)) # konversi ke int standar python
            confidence_score = float(np.max(probabilities)) # konversi ke float standar python

            # Ambil nama penyakit dari label yang sudah dimuat
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