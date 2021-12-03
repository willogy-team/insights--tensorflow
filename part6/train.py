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

from networks.vgg import VGG16Net
from networks.inceptionv3 import InceptionModule, InceptionNet


# To reproduce results
print(tf.__version__)
tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)


ap = argparse.ArgumentParser()
ap.add_argument("-trd", "--train_dir", required=True, help="Path to dataset train directory")
ap.add_argument("-td", "--test_dir", required=True, help="Path to dataset test directory")
ap.add_argument("-mdp", "--model_path", required=True, help="Path to the folder for saving checkpoints")
ap.add_argument("-imp", "--image_path", required=True, help="Path to the folder for saving images")
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

train_it = datagen.flow_from_directory(args["train_dir"], target_size=(224, 224), class_mode="categorical", batch_size=8)

test_it = datagen.flow_from_directory(args["test_dir"], target_size=(224, 224), class_mode="categorical", batch_size=8)

# confirm the iterator works
batchX, batchy = train_it.next()
print('[*] Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
print('[*] Batch shape=%s, min=%.3f, max=%.3f' % (batchy.shape, batchy.min(), batchy.max()))

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(8, 7, activation='relu'),
#     tf.keras.layers.Conv2D(8, 5, activation='relu'),
#     tf.keras.layers.Conv2D(8, 3, activation='relu'),
#     tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])

model = InceptionNet(num_classes=3)

input_shape = (None, 224, 224, 3)
model.build(input_shape)
model.model().summary()

image_path = args["image_path"]
os.makedirs(image_path, exist_ok=True)
tf.keras.utils.plot_model(model.model(), to_file="".join([image_path, "/model.png"]), show_shapes=True)

for layer in model.layers:
    # print('[*] layer: ', layer)
    if 'conv' not in layer.name:
        # print('No')
        continue

    filters_weights, biases_weights = layer.get_weights()
    # print('[**] layer.name: {}, filters_weights.shape: {}, biases_weights.shape: {}'.format(layer.name, filters_weights.shape, biases_weights.shape))
    filters_max, filters_min = filters_weights.max(), filters_weights.min()
    filters_weights = (filters_weights - filters_min)/(filters_max - filters_min)
    # print('[**] filters_weights: ', filters_weights)

# Note here between SparseCategoricalCrossentropy and categorical crossentropy
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

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

hist = model.fit(train_it, validation_data=test_it, epochs=100, callbacks=[tensorboard_callback, model_checkpoint_callback])

print("[*] Best validation accuracy: ", max(hist.history['val_accuracy']))
print("[*] Best validation loss: ", min(hist.history['val_loss']))

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

# probability_model = tf.keras.Sequential([model, 
#                                          tf.keras.layers.Softmax()])

# predictions = probability_model.predict(test_images)

# predictions[0]

# np.argmax(predictions[0])

# test_labels[0]