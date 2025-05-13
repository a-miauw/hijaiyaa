import streamlit as st
import numpy as np
import tensorflow as tf
import os, time
from PIL import Image
from playsound import playsound
import os
import gdown

# Cek  model 
MODEL_PATH = "model_hijaiyah.h5"
if not os.path.exists(MODEL_PATH):
    # Download dari Google Drive
    file_id = "19T6eQhsbIuMTNsmNNbcM0M6L7m-MrjAC"  # Ganti dengan ID kamu
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

#Load model
model = tf.keras.models.load_model('model_hijaiyah.h5')
classes = ['ain', 'alif', 'ba', 'dal', 'dhod', 'dzal', 'dzho', 'fa', 'ghoin', 'ha',
            'haa', 'jim', 'kah', 'kho', 'lam', 'mim', 'nun', 'qof', 'ro', 'shod', 
            'sin', 'syin', 'ta', 'tho', 'tsa', 'wawu', 'ya', 'zain'] 

st.set_page_config(page_title="Pengenalan Huruf Hijaiyah", layout="centered")
st.title("📸 Pengenalan Huruf Hijaiyah ")

#Ambil gambar dari kamera
img_file_buffer = st.camera_input("Ambil gambar tulisan tangan")

if img_file_buffer is not None:
    #Simpan dan proses gambar
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(img_file_buffer.getbuffer())

    img = Image.open(temp_path).convert("L").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)

    prediction = model.predict(img_array)
    pred_label = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if confidence >= 10:
        st.success(f"Huruf dikenali: {pred_label.upper()} (Akurasi: {confidence:.2f}%)")
        img.save(f"berhasil/{pred_label}_{timestamp}.jpg")

        #Suara
        sound_path = f"suara/{pred_label}.mp3"
        if os.path.exists(sound_path):
            playsound(sound_path)
    else:
        st.error(f"Akurasi terlalu rendah: {confidence:.2f}%")
        img.save(f"gagal/error_{timestamp}.jpg")
        if st.button("🔁 Ulangi"):
            st.rerun()
