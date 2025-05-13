import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Data augmentasi & normalisasi
datagen = ImageDataGenerator(
    rescale=1./255,              # Normalisasi pixel (0–255 jadi 0–1)
    rotation_range=20,           # Memutar gambar 
    width_shift_range=0.1,       # Menggeser gambar ke kanan/kiri max 10% lebar
    height_shift_range=0.1,      # Menggeser gambar ke atas/bawah max 10% tinggi
    shear_range=0.1,             # Melakukan transformasi miring 
    zoom_range=0.1,              # Memperbesar atau memperkecil gambar
    horizontal_flip=False,       # Tidak membalik gambar secara horizontal (karena huruf bisa jadi berubah makna)
    fill_mode='nearest',         # Metode mengisi area kosong setelah rotasi/pindah
    validation_split=0.2)

train_data = datagen.flow_from_directory(
    r'c:\laragon\www\hijaiyaa\dataset_hijaiyah',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    r'c:\laragon\www\hijaiyaa\dataset_hijaiyah',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'                                                                
)

#CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Training
model.fit(train_data, validation_data=val_data, epochs=100)

model.save('model_hijaiyah.h5')
print("✅ Model berhasil disimpan!")
