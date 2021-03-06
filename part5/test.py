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

from networks.vgg import VGG16Net

ap = argparse.ArgumentParser()
ap.add_argument("-trd", "--train_dir", required=True, help="Path to dataset train directory")
ap.add_argument("-mdp", "--model_path", required=True, help="Path to the folder for saving checkpoints")
args = vars(ap.parse_args())

model = VGG16Net(num_classes=3)
print('[*] type(model): ', type(model))
input_shape = (None, 224, 224, 3)
model.build(input_shape)
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
# model.summary()
model.model().summary()
checkpoint_path = os.path.join(args["model_path"], 'models')
model.load_weights(checkpoint_path)

for idx, layer in enumerate(model.layers):
    print('[*] layer: ', layer)
    if 'vgg_block' not in layer.name:
        print('No')
        continue
    print('[**] id: {}, layer.name: {}'.format(idx, layer.name))
    print('[**] len(layer.get_weights()): ', len(layer.get_weights()))
    list_of_weights = layer.get_weights()
    num_weights = len(list_of_weights)
    for i in range(int(num_weights/2)):
        print('[**] filters_weights.shape: {}, biases_weights.shape: {}'.format(
            list_of_weights[2*i].shape, # filter weights
            list_of_weights[2*i+1].shape) # bias weights
        )
        filters_max, filters_min = list_of_weights[2*i].max(), list_of_weights[2*i].min()
        filters_weights = (list_of_weights[2*i] - filters_min)/(filters_max - filters_min)
        # print('[**] filters_weights: ', filters_weights)
        plot_filters_of_a_layer(filters_weights, 10)
    print('[**] layer.output.shape: {}'.format(layer.output.shape))


# === Output feature maps from a single layer ===
# A PIL object
img = load_img(os.path.join(args["train_dir"], 'n02085620-Chihuahua', 'n02085620_1558.jpg'), target_size=(224, 224))
# Convert to numpy array
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
# img = model.preprocess_input(img)
img = img/255
print('[*] model, type(model): ', model, type(model))
print('[*] dir(model): ', dir(model))
# print('[*] model.input: ', model.input)
model.inputs = model.layers[0].input # !! VERY IMPORTANT
print('[*] model.inputs: ', model.inputs)
model.outputs = model.layers[-1].output # !! VERY IMPORTANT
print('[*] model.outputs: ', model.outputs)
# print('[*] model.input_shape: ', model.input_shape())
print('[*] model.layers: ', model.layers)
print('[*] model.layers[0]: ', model.layers[0])
print('[*] dir(model.layers[0]): ', dir(model.layers[0]))
print('[*] model.layers[0].input: ', model.layers[0].input)
print('[*] model.layers[0].output: ', model.layers[0].output)
x = tf.keras.layers.Input(shape=(224, 224, 3))
# model_1 = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
model_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
feature_maps_1 = model_1.predict(img)
print('[*] feature_maps_1.shape: ', feature_maps_1.shape)
    
plot_feature_maps_of_a_layer(feature_maps_1)
    
# === Output feature maps from multiple layers ===
list_of_outputs = [model.layers[idx].output for idx in range(5)]
model_2 = Model(inputs=model.layers[0].input, outputs=list_of_outputs)
model_2.summary()
feature_maps_2 = model_2.predict(img) 
for feature_map in feature_maps_2:
    print('[*] feature_map.shape: ', feature_map.shape)
    
plot_feature_maps_of_multiple_layers(feature_maps_2)
    
# === Output activation maximization from a single layer ===
plot_activation_maximization_of_a_layer(model, 4)
    
# === GradCam++ from a single layer ===
# plot_gradcam_plusplus_of_a_layer(model, 2)
    
# model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
model = model.model()
model.summary()
# === Attentions ===
image_titles = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog']
img1 = load_img(os.path.join(args["train_dir"], 'n02085620-Chihuahua', 'n02085620_1558.jpg'), target_size=(224, 224))
img2 = load_img(os.path.join(args["train_dir"], 'n02085782-Japanese_spaniel', 'n02085782_2874.jpg'), target_size=(224, 224))
img3 = load_img(os.path.join(args["train_dir"], 'n02085936-Maltese_dog', 'n02085936_4245.jpg'), target_size=(224, 224))
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