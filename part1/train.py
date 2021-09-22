import os
import argparse
import datetime
import shutil

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

ap = argparse.ArgumentParser()
ap.add_argument("-trd", "--train_dir", required=True, help="Path to dataset train directory")
ap.add_argument("-td", "--test_dir", required=True, help="Path to dataset test directory")
args = vars(ap.parse_args())

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

train_it = datagen.flow_from_directory(args["train_dir"], target_size=(128, 128), class_mode="categorical", batch_size=1)

test_it = datagen.flow_from_directory(args["test_dir"], target_size=(128, 128), class_mode="categorical", batch_size=1)

# confirm the iterator works
batchX, batchy = train_it.next()
print('[*] Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-7), metrics=['accuracy'])

if os.path.isdir("./logs"):
    shutil.rmtree("./logs") 
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_it, validation_data=test_it, epochs=50, callbacks=[tensorboard_callback])