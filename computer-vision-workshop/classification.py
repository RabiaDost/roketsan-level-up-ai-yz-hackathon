import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ayarlar 
DATA_DIR = "Dataset"
IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCH = 10

# veri setini eğitim ve validasyon olarak ayır
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split = 0.2,
    subset = "training",
    seed = 42,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

# sınıfları al
class_names = train_ds.class_names
print(f"class_name: {class_names}")

#veri yükleme
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# veri arttırma
data_augmentation = keras.Sequential([
layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0,1)])

#mobilinet modeli yükle
base_model = MobileNetV2(weights = "imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# yeni model
inputs = keras.Input(shape=(224,224,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

#compile
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

#modeli eğit
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCH)

# prediction
loss , accuracy = model.evaluate(val_ds)
print(f"accuracy: {accuracy}")
model.save("classifier.keras")