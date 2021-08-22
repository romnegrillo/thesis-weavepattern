import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model("cats_and_dogs_mobilenetv2.h5")
new_model.summary()


