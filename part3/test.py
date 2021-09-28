import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

from visualizations.manual_plot_by_matplotlib import plot_filters_of_a_layer
from visualizations.manual_plot_by_matplotlib import plot_feature_maps_of_a_layer, plot_feature_maps_of_multiple_layers
from visualizations.automatic_plot_by_tf_keras_vis import plot_activation_maximization_of_a_layer
from visualizations.automatic_plot_by_tf_keras_vis import plot_vanilla_saliency_of_a_model
from visualizations.automatic_plot_by_tf_keras_vis import plot_smoothgrad_of_a_model
from visualizations.automatic_plot_by_tf_keras_vis import plot_gradcam_of_a_model
from visualizations.automatic_plot_by_tf_keras_vis import plot_gradcam_plusplus_of_a_model
from visualizations.automatic_plot_by_tf_keras_vis import plot_scorecam_of_a_model
from visualizations.automatic_plot_by_tf_keras_vis import plot_faster_scorecam_of_a_model


def create_model():
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

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model

model = create_model()
checkpoint_path = './models/models'
model.load_weights(checkpoint_path)

for idx, layer in enumerate(model.layers):
    print('[*] layer: ', layer)
    if 'conv' not in layer.name:
        print('No')
        continue

    filters_weights, biases_weights = layer.get_weights()
    print('[**] id: {}, layer.name: {}, filters_weights.shape: {}, biases_weights.shape: {}'.format(idx, layer.name, filters_weights.shape, biases_weights.shape))
    print('[**] layer.output.shape: {}'.format(layer.output.shape))
    filters_max, filters_min = filters_weights.max(), filters_weights.min()
    filters_weights = (filters_weights - filters_min)/(filters_max - filters_min)
    # print('[**] filters_weights: ', filters_weights)

    plot_filters_of_a_layer(filters_weights, 3)

# === Output feature maps from a single layer ===
# A PIL object
img = load_img('/media/data-huy/dataset/StanfordDogs/train_val_test/train/n02085620-Chihuahua/n02085620_1558.jpg', target_size=(128, 128))
# Convert to numpy array
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
# img = model.preprocess_input(img)
img = img/255
model_1 = Model(inputs=model.inputs, outputs=model.layers[0].output)
feature_maps_1 = model_1.predict(img)
print('[*] feature_maps_1.shape: ', feature_maps_1.shape)

plot_feature_maps_of_a_layer(feature_maps_1)

# === Output feature maps from multiple layers ===
list_of_outputs = [model.layers[idx].output for idx in range(3)]
model_2 = Model(inputs=model.inputs, outputs=list_of_outputs)
model_2.summary()
feature_maps_2 = model_2.predict(img) 
for feature_map in feature_maps_2:
    print('[*] feature_map.shape: ', feature_map.shape)

plot_feature_maps_of_multiple_layers(feature_maps_2)

# === Output activation maximization from a single layer ===
plot_activation_maximization_of_a_layer(model, 2)

# === GradCam++ from a single layer ===
# plot_gradcam_plusplus_of_a_layer(model, 2)

# === Attentions ===
image_titles = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog']
img1 = load_img('/media/data-huy/dataset/StanfordDogs/train_val_test/train/n02085620-Chihuahua/n02085620_1558.jpg', target_size=(128, 128))
img2 = load_img('/media/data-huy/dataset/StanfordDogs/train_val_test/train/n02085782-Japanese_spaniel/n02085782_2874.jpg', target_size=(128, 128))
img3 = load_img('/media/data-huy/dataset/StanfordDogs/train_val_test/train/n02085936-Maltese_dog/n02085936_4245.jpg', target_size=(128, 128))
img1 = np.asarray(img1)
img2 = np.asarray(img2)
img3 = np.asarray(img3)
images = np.asarray([img1, img2, img3])

X = images/255

## Vanilla saliency
print('[*] Vanilla saliency')
plot_vanilla_saliency_of_a_model(model, X, image_titles)

## SmoothGrad
print('[*] SmoothGrad')
plot_smoothgrad_of_a_model(model, X, image_titles)

## GradCAM
print('[*] GradCAM')
plot_gradcam_of_a_model(model, X, image_titles, images)

## GradCAM++
print('[*] GradCAM++')
plot_gradcam_plusplus_of_a_model(model, X, image_titles, images)

## ScoreCAM
print('[*] ScoreCam')
plot_scorecam_of_a_model(model, X, image_titles, images)

## Faster-ScoreCAM
print('[*] Faster-ScoreCAM')
plot_faster_scorecam_of_a_model(model, X, image_titles, images)