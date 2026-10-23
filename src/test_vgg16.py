import tensorflow as tf
from tensorflow.keras.applications import VGG16

print("TensorFlow version:", tf.__version__)

model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

print("\nVGG16 loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Total layers:", len(model.layers))
print("Trainable parameters:", model.count_params())
