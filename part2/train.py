import os
import argparse
import datetime
import shutil
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)
tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)

ap = argparse.ArgumentParser()
ap.add_argument("-trd", "--train_dir", required=True, help="Path to dataset train directory")
ap.add_argument("-td", "--test_dir", required=True, help="Path to dataset test directory")
ap.add_argument("-mdp", "--model_path", required=True, help="Path to the folder for saving checkpoints")
ap.add_argument("-imp", "--image_path", required=True, help="Path to the folder for saving the image of model plot")
args = vars(ap.parse_args())

datagen = ImageDataGenerator(
    rescale=1./255,
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
    tf.keras.layers.Conv2D(8, 7, activation='relu'),
    tf.keras.layers.Conv2D(8, 5, activation='relu'),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
input_shape = (None, 128, 128, 3)
model.build(input_shape)
model.summary()
image_path = args["image_path"]
os.makedirs(image_path, exist_ok=True)
tf.keras.utils.plot_model(model, to_file="".join([image_path, "/model.png"]), show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

if os.path.isdir("./logs"):
    shutil.rmtree("./logs") 
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

os.makedirs(args["model_path"], exist_ok=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="".join([args["model_path"], "/models"]),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

hist = model.fit(train_it, validation_data=test_it, epochs=50, callbacks=[tensorboard_callback, model_checkpoint_callback])

print("[*] Best validation accuracy: ", max(hist.history['val_accuracy']))
print("[*] Best validation loss: ", min(hist.history['val_loss']))