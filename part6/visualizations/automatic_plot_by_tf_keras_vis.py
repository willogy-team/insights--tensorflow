import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


def plot_activation_maximization_of_a_layer(model, layer_index):
    # Create the visualization instance.
    # All visualization classes accept a model and model-modifier, which, for example,
    #     replaces the activation of last layer to linear function so on, in constructor.
    activation_maximization = \
    ActivationMaximization(model,
                            model_modifier=[ExtractIntermediateLayer(model.layers[layer_index].name)],
                            clone=False)

    # You can use Score class to specify visualizing target you want.
    # And add regularizers or input-modifiers as needed.
    FILTER_INDEX = 500
    # activations = \
    # activation_maximization(CategoricalScore(FILTER_INDEX),
    #                         steps=100,
    #                         input_modifiers=[Jitter(jitter=16), Rotate2D(degree=1)],
    #                         regularizers=[TotalVariation2D(weight=1.0),
    #                                         Norm(weight=0.3, p=1)],
    #                         optimizer=tf.keras.optimizers.Adam(1.0, 0.999),
    #                         callbacks=[Progress()])
    activations = \
    activation_maximization(CategoricalScore(FILTER_INDEX),
                            steps=500,
                            callbacks=[Progress()])

    ## Since v0.6.0, calling `astype()` is NOT necessary.
    # activations = activations[0].astype(np.uint8)

    # Render
    # print('[*] activations: ', activations)
    print('[*] activations.shape: ', activations.shape)
    plt.imshow(activations[0])
    plt.show()

def plot_vanilla_saliency_of_a_model(model, X, image_titles):
    score = CategoricalScore(list(range(X.shape[0])))

    # === Vanilla Saliency ===
    # Create Saliency object
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Generate saliency map
    saliency_map = saliency(score, X)

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_smoothgrad_of_a_model(model, X, image_titles):
    score = CategoricalScore(list(range(X.shape[0])))

    # === Vanilla Saliency ===
    # Create Saliency object
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Generate saliency map
    saliency_map = saliency(score, X,
                            smooth_samples=20, # The number of calculating gradients iterations
                            smooth_noise=0.20) # noise spread level

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_gradcam_of_a_model(model, X, image_titles, images):
    score = CategoricalScore(list(range(X.shape[0])))
    # filter_numbers = [63, 132, 320]
    # score = CategoricalScore(filter_numbers)

    # Create Gradcam object
    gradcam = Gradcam(model,
                     model_modifier=ReplaceToLinear(),
                     clone=True)

    # Generate heatmap with GradCAM
    cam = gradcam(score,
                  X,
                  seek_penultimate_conv_layer=False,
                  penultimate_layer='vgg_block_4')

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_gradcam_plusplus_of_a_model(model, X, image_titles, images):
    score = CategoricalScore(list(range(X.shape[0])))
    # filter_numbers = [63, 132, 320]
    # score = CategoricalScore(filter_numbers)
    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(model,
                            model_modifier=ReplaceToLinear(),
                            clone=True)

    # Generate heatmap with GradCAM++
    cam = gradcam(score,
                  X,
                  seek_penultimate_conv_layer=False,
                  penultimate_layer='vgg_block_4')

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_scorecam_of_a_model(model, X, image_titles, images):
    score = CategoricalScore(list(range(X.shape[0])))
    # filter_numbers = [63, 132, 320]
    # score = CategoricalScore(filter_numbers)
    # Create ScoreCAM object
    scorecam = Scorecam(model, 
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Generate heatmap with ScoreCAM
    cam = scorecam(score, 
                   X, 
                   seek_penultimate_conv_layer=False,
                   penultimate_layer='vgg_block_4')

    ## Since v0.6.0, calling `normalize()` is NOT necessary.
    # cam = normalize(cam)

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_faster_scorecam_of_a_model(model, X, image_titles, images):
    score = CategoricalScore(list(range(X.shape[0])))
    # Create ScoreCAM object
    scorecam = Scorecam(model, 
                        model_modifier=ReplaceToLinear(),
                        clone=True)

    # Generate heatmap with Faster-ScoreCAM
    cam = scorecam(score,
                   X,
                   seek_penultimate_conv_layer=False, 
                   penultimate_layer='vgg_block_4',
                   max_N=8)

    ## Since v0.6.0, calling `normalize()` is NOT necessary.
    # cam = normalize(cam)

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

