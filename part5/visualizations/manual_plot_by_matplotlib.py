import matplotlib.pyplot as plt
import numpy as np


def plot_filters_of_a_layer(filters_weights, num_filters):
    # plot first few filters
    ix = 1
    for i in range(num_filters):
        # get the filter
        f = filters_weights[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(num_filters, 3, ix)
            # filters = ['filter 1', 'filter 2', 'filter 3']
            # channels = ['channel 1', ' channel 2', 'channel 3']
            # ax.set_xticks(np.arange(len(filters)))
            # ax.set_yticks(np.arange(len(channels)))
            # ax.set_xtickslabels(filters)
            # ax.set_ytickslabels(channels)
            ax.set_xticks([])
            ax.set_yticks([])

            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

def plot_feature_maps_of_a_layer(feature_maps):
    ix = 1
    for _ in range(2):
        for _ in range(4):
            ax = plt.subplot(2, 4, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1

    plt.show()

def plot_feature_maps_of_multiple_layers(feature_maps):
    for feature_map in feature_maps:
        plot_feature_maps_of_a_layer(feature_map)