import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Set page config
st.set_page_config(
    page_title="PakDe - Parkinson Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4a6cf7;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
    }
    .info-text {
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #4a6cf7;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3a5bd9;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pakde_folder_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image
def preprocess_image(img, target_size=(224, 224)):
    # Convert to RGB
    img = img.convert('RGB')
    # Resize image
    img = img.resize(target_size)
    # Convert to array
    img_array = np.array(img).astype(np.float32)
    # Preprocessing: normalize to [-1,1] range
    img_array = (img_array / 127.5) - 1.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main function
def main():
    # Langsung tampilkan halaman home tanpa sidebar navigasi
    show_home()
    
    # Tampilkan footer
    show_footer()

# Home page
def show_home():
    st.title("Deteksi Parkinson")
    st.write("Sistem deteksi penyakit Parkinson berbasis kecerdasan buatan melalui analisis tulisan tangan")
    
    st.header("Unggah Gambar Tulisan Tangan")
    
    # File uploader
    uploaded_file = st.file_uploader("Pilih Gambar (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        # Predict button
        predict_button = st.button("Analisis Gambar", use_container_width=True)
        
        if predict_button:
            with st.spinner("Menganalisis gambar..."):
                # Load model
                model = load_model()
                
                if model is not None:
                    # Preprocess image
                    processed_img = preprocess_image(image)
                    
                    # Make prediction
                    prediction = model.predict(processed_img)
                    prediction_value = float(prediction[0][0])
                    
                    # Determine result
                    if prediction_value > 0.5:
                        result = "Parkinson Terdeteksi"
                        icon = "⚠️"
                    else:
                        result = "Tidak Terdeteksi Parkinson"
                        icon = "✅"
                    
                    # Calculate confidence
                    confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
                    confidence_percentage = round(confidence * 100, 2)
                    
                    # Display result
                    st.subheader("Hasil Deteksi")
                    st.markdown(f"### {icon} {result}")
                    st.write(f"Tingkat Kepercayaan: {confidence_percentage}%")
                    st.progress(confidence / 1.0)
                    
                    st.info(
                        "**Catatan Penting**: Hasil ini hanya bersifat indikatif dan tidak menggantikan diagnosis medis profesional. "
                        "Silakan konsultasikan dengan dokter untuk evaluasi lebih lanjut. Jika Anda memiliki kekhawatiran tentang gejala Parkinson, "
                        "segera hubungi profesional kesehatan."
                    )
    
    # Tidak ada konten tambahan yang perlu ditampilkan

# Footer
def show_footer():
    st.markdown("---")
    st.caption("© 2025 PakDe - Parkinson Detection. Aplikasi ini hanya untuk tujuan demonstrasi.")

# Run the app
if __name__ == "__main__":
    main()