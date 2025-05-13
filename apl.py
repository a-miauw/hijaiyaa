import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Load Model dan Data
model = load_model('model_hijaiyah.h5')

# Augmentasi dan Normalisasi Data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

# Load data validasi
val_data = datagen.flow_from_directory(
    'dataset_hijaiyah',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Menyimpan hasil training untuk grafik
history = model.history

#  Menampilkan Grafik Akurasi dan Loss
st.title(" Grafik Pelatihan Model CNN")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Akurasi
ax[0].plot(history.history['accuracy'], label='Training Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Akurasi Model')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Akurasi')
ax[0].legend()

# Loss
ax[1].plot(history.history['loss'], label='Training Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Loss Model')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

st.pyplot(fig)

# Prediksi Gambar
st.title(" Prediksi Gambar Huruf Hijaiyah")

x, y = next(val_data)
predictions = model.predict(x)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y, axis=1)
class_labels = list(val_data.class_indices.keys())

# Menampilkan 9 gambar pertama
fig2, ax2 = plt.subplots(3, 3, figsize=(10, 10))

for i in range(9):
    ax2[i//3, i%3].imshow(x[i].reshape(224, 224), cmap='gray')
    ax2[i//3, i%3].set_title(f"Pred: {class_labels[predicted_classes[i]]}\nTrue: {class_labels[true_classes[i]]}")
    ax2[i//3, i%3].axis('off')

st.pyplot(fig2)

# Menambahkan tombol untuk memulai ulang prediksi
if st.button('Prediksi Ulang Gambar'):
    st.rerun()
