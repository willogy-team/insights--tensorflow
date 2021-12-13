import numpy as np
import tensorflow as tf


class InceptionNet(tf.keras.Model):
    def __init__(self, num_classes=3):
        super(InceptionNet, self).__init__()
        # self.inception_module_1 = InceptionModule(32, 32, 64, 16, 32, 32)
        # self.inception_module_2 = InceptionModule(32, 64, 128, 32, 64, 16)
        self.inception_module_1 = InceptionModule(64, 96, 128, 16, 32, 32)
        self.inception_module_2 = InceptionModule(128, 128, 32, 32, 32, 16)
        self.flatten = tf.keras.layers.Flatten(input_shape=(3, 3, 10))
        self.dense_1 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(9, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(9, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.inception_module_1(inputs)
        x = self.inception_module_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.classifier(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class InceptionModuleWithFactorization(tf.keras.layers.Layer):
    '''
    Inception module with asymmetric factorization
    '''
    def __init__(self, nf1, nf2_a, nf2_b, nf3_a, nf3_b, nf4, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        self.n_filters_of_conv_layer_1 = nf1
        self.n_filters_of_conv_layer_2_a = nf2_a
        self.n_filters_of_conv_layer_2_b = nf2_b
        self.n_filters_of_conv_layer_3_a = nf3_a
        self.n_filters_of_conv_layer_3_b = nf3_b
        self.n_filters_of_conv_layer_4 = nf4

    def build(self, input_shape):
        # === Path for the 1x1 convolutional layer ===
        self.conv2d_1_nf1 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_1,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        # === Path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf2_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                       (3, 1),
                                                       padding='same',
                                                       activation='relu')
        self.conv2d_3_nf2_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                        (1, 3),
                                                        padding='same',
                                                        activation='relu')
        # === Path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf3_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                       (3, 1),
                                                       padding='same',
                                                       activation='relu')
        self.conv2d_3_nf3_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                        (1, 3),
                                                        padding='same',
                                                        activation='relu')
        self.conv2d_3_nf3_b_iii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                         (3, 1),
                                                         padding='same',
                                                         activation='relu')
        self.conv2d_3_nf3_b_iv = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                        (1, 3),
                                                        padding='same',
                                                        activation='relu')
        # === Path for the 3x3 max-pool layer ===
        self.max_pool2d = tf.keras.layers.MaxPool2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
                                                   
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === Path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)

        # === Path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b_i = self.conv2d_3_nf2_b_i(conv2d_1_nf2_a)
        conv2d_3_nf2_b_ii = self.conv2d_3_nf2_b_ii(conv2d_3_nf2_b_i)

        # === Path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(conv2d_1_nf3_a)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(conv2d_3_nf3_b_i)
        conv2d_3_nf3_b_iii = self.conv2d_3_nf3_b_iii(conv2d_3_nf3_b_ii)
        conv2d_3_nf3_b_iv = self.conv2d_3_nf3_b_iv(conv2d_3_nf3_b_iii)

        # === Path for the 3x3 max-pool layer ===
        max_pool2d = self.max_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(max_pool2d)

        # === Concatenation ===
        concatenation = self.concatenation([conv2d_1_nf1, 
                                            conv2d_3_nf2_b_ii, 
                                            conv2d_3_nf3_b_iv, 
                                            conv2d_1_nf4])

        return concatenation
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters_of_conv_layer_1': self.n_filters_of_conv_layer_1,
            'n_filters_of_conv_layer_2_a': self.n_filters_of_conv_layer_2_a,
            'n_filters_of_conv_layer_2_b': self.n_filters_of_conv_layer_2_b,
            'n_filters_of_conv_layer_3_a': self.n_filters_of_conv_layer_3_a,
            'n_filters_of_conv_layer_3_b': self.n_filters_of_conv_layer_3_b,
            'n_filters_of_conv_layer_4': self.n_filters_of_conv_layer_4,
        })

        return config

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class InceptionModuleForHighDimRepresentations(tf.keras.layers.Layer):
    def __init__(self, nf1, nf2_a, nf2_b, nf3_a, nf3_b, nf4, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        self.n_filters_of_conv_layer_1 = nf1
        self.n_filters_of_conv_layer_2_a = nf2_a
        self.n_filters_of_conv_layer_2_b = nf2_b
        self.n_filters_of_conv_layer_3_a = nf3_a
        self.n_filters_of_conv_layer_3_b = nf3_b
        self.n_filters_of_conv_layer_4 = nf4

    def build(self, input_shape):
        # === Path for the 1x1 convolutional layer ===
        self.conv2d_1_nf1 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_1,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        # === Path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf2_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                       (3, 1),
                                                       padding='same',
                                                       activation='relu')
        self.conv2d_3_nf2_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                        (1, 3),
                                                        padding='same',
                                                        activation='relu')
        # === Path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf3_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                       (3, 3),
                                                       padding='same',
                                                       activation='relu')
        self.conv2d_3_nf3_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                        (3, 1),
                                                        padding='same',
                                                        activation='relu')
        self.conv2d_3_nf3_b_iii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                         (1, 3),
                                                         padding='same',
                                                         activation='relu')
        # === Path for the 3x3 max-pool layer ===
        self.max_pool2d = tf.keras.layers.MaxPool2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === Path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)

        # === Path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b_i = self.conv2d_3_nf2_b_i(conv2d_1_nf2_a)
        conv2d_3_nf2_b_ii = self.conv2d_3_nf2_b_ii(conv2d_1_nf2_a)

        # === Path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(conv2d_1_nf3_a)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(conv2d_3_nf3_b_i)
        conv2d_3_nf3_b_iii = self.conv2d_3_nf3_b_iii(conv2d_3_nf3_b_i)

        # === Path for the 3x3 max-pool layer ===
        max_pool2d = self.max_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(max_pool2d)

        # === Concatenation ===
        concatenation = self.concatenation([conv2d_1_nf1, 
                                            conv2d_3_nf2_b_i, 
                                            conv2d_3_nf2_b_ii, 
                                            conv2d_3_nf3_b_ii, 
                                            conv2d_3_nf3_b_iii, 
                                            conv2d_1_nf4])

        return concatenation
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters_of_conv_layer_1': self.n_filters_of_conv_layer_1,
            'n_filters_of_conv_layer_2_a': self.n_filters_of_conv_layer_2_a,
            'n_filters_of_conv_layer_2_b': self.n_filters_of_conv_layer_2_b,
            'n_filters_of_conv_layer_3_a': self.n_filters_of_conv_layer_3_a,
            'n_filters_of_conv_layer_3_b': self.n_filters_of_conv_layer_3_b,
            'n_filters_of_conv_layer_4': self.n_filters_of_conv_layer_4,
        })

        return config

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, nf1, nf2_a, nf2_b, nf3_a, nf3_b, nf4, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        self.n_filters_of_conv_layer_1 = nf1
        self.n_filters_of_conv_layer_2_a = nf2_a
        self.n_filters_of_conv_layer_2_b = nf2_b
        self.n_filters_of_conv_layer_3_a = nf3_a
        self.n_filters_of_conv_layer_3_b = nf3_b
        self.n_filters_of_conv_layer_4 = nf4

    def build(self, input_shape):
        # === Path for the 1x1 convolutional layer ===
        self.conv2d_1_nf1 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_1,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        # === Path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf2_b = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        # === Path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_5_nf3_b = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                     (5, 5),
                                                     padding='same',
                                                     activation='relu')
        # === Path for the 3x3 max-pool layer ===
        self.max_pool2d = tf.keras.layers.MaxPool2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === Path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)

        # === Path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b = self.conv2d_3_nf2_b(conv2d_1_nf2_a)

        # === Path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        conv2d_5_nf3_b = self.conv2d_5_nf3_b(conv2d_1_nf3_a)

        # === Path for the 3x3 max-pool layer ===
        max_pool2d = self.max_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(max_pool2d)

        # === Concatenation ===
        concatenation = self.concatenation([conv2d_1_nf1, 
                                            conv2d_3_nf2_b, 
                                            conv2d_5_nf3_b, 
                                            conv2d_1_nf4])

        return concatenation
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters_of_conv_layer_1': self.n_filters_of_conv_layer_1,
            'n_filters_of_conv_layer_2_a': self.n_filters_of_conv_layer_2_a,
            'n_filters_of_conv_layer_2_b': self.n_filters_of_conv_layer_2_b,
            'n_filters_of_conv_layer_3_a': self.n_filters_of_conv_layer_3_a,
            'n_filters_of_conv_layer_3_b': self.n_filters_of_conv_layer_3_b,
            'n_filters_of_conv_layer_4': self.n_filters_of_conv_layer_4,
        })

        return config

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))