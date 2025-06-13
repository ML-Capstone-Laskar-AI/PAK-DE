import os
import time
from flask import Flask, render_template, request, jsonify
# Menggunakan TensorFlow untuk model Keras
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load Keras model
model_path = os.path.join(os.path.dirname(__file__), 'pakde_folder_model.keras')
model = tf.keras.models.load_model(model_path)

# Fungsi untuk preprocessing gambar
def preprocess_image(img, target_size=(224, 224)):
    # Konversi ke RGB
    img = img.convert('RGB')
    # Resize gambar
    img = img.resize(target_size)
    # Konversi ke array
    img_array = np.array(img).astype(np.float32)

    # Preprocessing MobileNetV2: mengubah rentang piksel [0,255] menjadi [-1,1]
    # Menggunakan operasi NumPy manual untuk kompatibilitas
    img_array = (img_array / 127.5) - 1.0

    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('home.html')

# Route untuk halaman tentang penyakit Parkinson
@app.route('/about')
def about():
    return render_template('about.html')

# Route untuk halaman jurnal referensi
@app.route('/journal')
def journal():
    return render_template('journal.html')

# Route untuk halaman tim capstone
@app.route('/team')
def team():
    return render_template('team.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'})
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        prediction_value = float(prediction[0][0])
        result = "Parkinson Terdeteksi" if prediction_value > 0.5 else "Sehat"
        confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
        confidence_percentage = round(confidence * 100, 2)
        elapsed = time.time() - start_time
        print(f"Prediction took {elapsed:.3f} seconds")
        return jsonify({
            'result': result,
            'confidence': confidence_percentage
        })
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Prediction failed after {elapsed:.3f} seconds")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

