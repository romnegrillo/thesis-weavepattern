import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model("cloth_pattern_mobilenetv2.h5")
new_model.summary()


