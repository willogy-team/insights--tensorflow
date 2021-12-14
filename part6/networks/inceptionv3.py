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
        # === First path for the 1x1 convolutional layer ===
        self.conv2d_1_nf1 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_1,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        # === Second path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf2_b = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        # === Third path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_5_nf3_b = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                     (5, 5),
                                                     padding='same',
                                                     activation='relu')
        # === Fourth path for the 3x3 max-pool layer ===
        self.avg_pool2d = tf.keras.layers.AveragePooling2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)

        # === Second path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b = self.conv2d_3_nf2_b(conv2d_1_nf2_a)

        # === Third path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        conv2d_5_nf3_b = self.conv2d_5_nf3_b(conv2d_1_nf3_a)

        # === Fourth path for the 3x3 max-pool layer ===
        avg_pool2d = self.avg_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(avg_pool2d)

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

class InceptionModuleWithFactorization(tf.keras.layers.Layer):
    '''
    Inception module with asymmetric factorization
    '''
    def __init__(self, nf1, nf2_a, nf2_b, nf3_a, nf3_b, nf4, **kwargs):
        super(InceptionModuleWithFactorization, self).__init__(**kwargs)
        self.n_filters_of_conv_layer_1 = nf1
        self.n_filters_of_conv_layer_2_a = nf2_a
        self.n_filters_of_conv_layer_2_b = nf2_b
        self.n_filters_of_conv_layer_3_a = nf3_a
        self.n_filters_of_conv_layer_3_b = nf3_b
        self.n_filters_of_conv_layer_4 = nf4

    def build(self, input_shape):
        # === First path for the 1x1 convolutional layer ===
        self.conv2d_1_nf1 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_1,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        # === Second path for the 3x3 convolutional layer ===
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
        # === Third path for the 5x5 convolutional layer ===
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
        # === Fourth path for the 3x3 max-pool layer ===
        self.avg_pool2d = tf.keras.layers.AveragePooling2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')

        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)

        # === Second path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b_i = self.conv2d_3_nf2_b_i(conv2d_1_nf2_a)
        conv2d_3_nf2_b_ii = self.conv2d_3_nf2_b_ii(conv2d_3_nf2_b_i)

        # === Third path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(conv2d_1_nf3_a)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(conv2d_3_nf3_b_i)
        conv2d_3_nf3_b_iii = self.conv2d_3_nf3_b_iii(conv2d_3_nf3_b_ii)
        conv2d_3_nf3_b_iv = self.conv2d_3_nf3_b_iv(conv2d_3_nf3_b_iii)

        # === Fourth path for the 3x3 max-pool layer ===
        avg_pool2d = self.avg_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(avg_pool2d)

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
        super(InceptionModuleForHighDimRepresentations, self).__init__(**kwargs)
        self.n_filters_of_conv_layer_1 = nf1
        self.n_filters_of_conv_layer_2_a = nf2_a
        self.n_filters_of_conv_layer_2_b = nf2_b
        self.n_filters_of_conv_layer_3_a = nf3_a
        self.n_filters_of_conv_layer_3_b = nf3_b
        self.n_filters_of_conv_layer_4 = nf4

    def build(self, input_shape):
        # === First path for the 1x1 convolutional layer ===
        self.conv2d_1_nf1 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_1,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        # === Second path for the 3x3 convolutional layer ===
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
        # === Third path for the 5x5 convolutional layer ===
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
        # === Fourth path for the 3x3 max-pool layer ===
        self.avg_pool2d = tf.keras.layers.AveragePooling2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')

        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)

        # === Second path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b_i = self.conv2d_3_nf2_b_i(conv2d_1_nf2_a)
        conv2d_3_nf2_b_ii = self.conv2d_3_nf2_b_ii(conv2d_1_nf2_a)

        # === Third path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(conv2d_1_nf3_a)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(conv2d_3_nf3_b_i)
        conv2d_3_nf3_b_iii = self.conv2d_3_nf3_b_iii(conv2d_3_nf3_b_i)

        # === Fourth path for the 3x3 max-pool layer ===
        avg_pool2d = self.avg_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(avg_pool2d)

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

class AuxiliaryClassifier(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AuxiliaryClassifier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_5_a = tf.keras.layers.Conv2D(768,
                                                 (5, 5),
                                                 padding='same',
                                                 activation='relu')
        self.conv2d_5_b = tf.keras.layers.Conv2D(128,
                                                 (5, 5),
                                                 padding='same',
                                                 activation='relu')
        self.dense = tf.keras.layers.Dense(1024, activation='relu')

    def call(self, input_tensor, training=False):
        conv2d_5_a = self.conv2d_5_a(input_tensor)
        conv2d_5_b = self.conv2d_5_b(conv2d_5_a)
        dense = self.dense(conv2d_5_b)

        return dense

    def get_config(self):
        config = super().get_config().copy()
        return config

    def model(self):
        x = tf.keras.layers.Input(shape=(17, 17, 768))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class GridSizeReduction(tf.keras.layers.Layer):
    def __init__(self, nf1, nf2, nf3, **kwargs):
        super(GridSizeReduction, self).__init__(**kwargs)
        self.n_filters_of_layer_1 = nf1
        self.n_filters_of_layer_2 = nf2
        self.n_filters_of_layer_3 = nf3

    def build(self, input_shape):
        # === First path ===
        self.conv2d_1_nf1_a = tf.keras.layers.Conv2D(self.n_filters_of_layer_1,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf1_b = tf.keras.layers.Conv2D(self.n_filters_of_layer_1,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf1_c = tf.keras.layers.Conv2D(self.n_filters_of_layer_1,
                                                     (3, 3),
                                                     strides=(2, 2),
                                                     padding='same',
                                                     activation='relu')

        # === Second path ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_layer_2,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.conv2d_3_nf2_b = tf.keras.layers.Conv2D(self.n_filters_of_layer_2,
                                                     (3, 3),
                                                     strides=(2, 2),
                                                     padding='same',
                                                     activation='relu')

        # === Third path ===
        self.max_pool2d = tf.keras.layers.MaxPool2D((17, 17),
                                                     strides=(2, 2),
                                                     padding='same')

        # === Concatenation ===
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path ===
        conv2d_1_nf1_a = self.conv2d_1_nf1_a(input_tensor)
        conv2d_3_nf1_b = self.conv2d_3_nf1_b(conv2d_1_nf1_a)
        conv2d_3_nf1_c = self.conv2d_3_nf1_c(conv2d_3_nf1_b)

        # === Second path ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        conv2d_3_nf2_b = self.conv2d_1_nf2_b(conv2d_1_nf2_a)

        # === Third path ===
        max_pool2d = self.max_pool2d(input_tensor)

        # === Concatenation ===
        concatenation = self.concatenation([conv2d_3_nf1_c,
                                            conv2d_3_nf2_b,
                                            max_pool2d])
        return concatenation

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_filters_of_layer_1': self.n_filters_of_layer_1,
            'n_filters_of_layer_2': self.n_filters_of_layer_2,
            'n_filters_of_layer_3': self.n_filters_of_layer_3,
        })

        return config

    def model(self):
        x = tf.keras.layers.Input(shape=(35, 35, 320))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))