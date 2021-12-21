import numpy as np
import tensorflow as tf


class InceptionNet(tf.keras.Model):
    def __init__(self, num_classes=3):
        super(InceptionNet, self).__init__()
        self.conv2d_a_1 = tf.keras.layers.Conv2D(filters=32,
                                                 kernel_size=(3, 3),
                                                 strides=(2, 2),
                                                 padding='valid',
                                                 activation=None)
        self.conv2d_a_2 = tf.keras.layers.Conv2D(filters=32,
                                                 kernel_size=(3, 3),
                                                 padding='valid',
                                                 activation=None)
        self.conv2d_a_3 = tf.keras.layers.Conv2D(filters=64,
                                                 kernel_size=(3, 3),
                                                 padding='same',
                                                 activation=None)
        self.max_pool2d_a = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                      strides=(2, 2),
                                                      padding='valid')
        self.conv2d_b_1 = tf.keras.layers.Conv2D(filters=80,
                                                 kernel_size=(1, 1),
                                                 padding='valid',
                                                 activation=None)
        self.conv2d_b_2 = tf.keras.layers.Conv2D(filters=192,
                                                 kernel_size=(3, 3),
                                                 padding='valid',
                                                 activation=None)
        self.max_pool2d_b = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                      strides=(2, 2),
                                                      padding='valid')
        self.inception_module_a_1 = InceptionModule(64, 48, 64, 64, 96, 32)
        self.inception_module_a_2 = InceptionModule(64, 48, 64, 64, 96, 64)
        self.inception_module_a_3 = InceptionModule(64, 48, 64, 64, 96, 64)
        self.grid_size_reduction_1 = GridSizeReduction(64, 96, 384, 384)
        self.inception_module_b_1 = InceptionModuleWithAsymmetricFactorization(192, 128, 192, 128, 192, 192)
        self.inception_module_b_2 = InceptionModuleWithAsymmetricFactorization(192, 160, 192, 160, 192, 192)
        self.inception_module_b_3 = InceptionModuleWithAsymmetricFactorization(192, 160, 192, 160, 192, 192)
        self.inception_module_b_4 = InceptionModuleWithAsymmetricFactorization(192, 192, 192, 192, 192, 192)
        self.auxiliary_classifier = AuxiliaryClassifier()
        self.grid_size_reduction_2 = GridSizeReduction(192, 192, 192, 320)
        self.inception_module_c_1 = InceptionModuleForHighDimRepresentations(320, 384, 384, 448, 384, 192)
        self.inception_module_c_2 = InceptionModuleForHighDimRepresentations(320, 384, 384, 448, 384, 192)
        self.average_pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                            strides=None,
                                                            padding='valid')
        self.dropout = tf.keras.layers.Dropout(rate=0.2,
                                               noise_shape=None,
                                               seed=None)
        self.flatten = tf.keras.layers.Flatten(input_shape=(3, 3, 10))
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv2d_a_1(inputs)
        x = self.conv2d_a_2(x)
        x = self.conv2d_a_3(x)
        x = self.max_pool2d_a(x)
        x = self.conv2d_b_1(x)
        x = self.conv2d_b_2(x)
        x = self.max_pool2d_b(x)
        x = self.inception_module_a_1(x)
        x = self.inception_module_a_2(x)
        x = self.inception_module_a_3(x)
        x = self.grid_size_reduction_1(x)
        x = self.inception_module_b_1(x)
        x = self.inception_module_b_2(x)
        x = self.inception_module_b_3(x)
        x = self.inception_module_b_4(x)
        out1 = self.auxiliary_classifier(x) 
        x = self.grid_size_reduction_2(x)
        x = self.inception_module_c_1(x)
        x = self.inception_module_c_2(x)
        x = self.average_pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        out2 = self.classifier(x)

        return [out1, out2]

    def model(self):
        x = tf.keras.layers.Input(shape=(299, 299, 3))
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
        self.bn_1_nf1 = tf.keras.layers.BatchNormalization()
        # === Second path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf2_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf2_b = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        self.bn_3_nf2_b = tf.keras.layers.BatchNormalization()
        # === Third path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf3_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        self.bn_3_nf3_b_i = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        self.bn_3_nf3_b_ii = tf.keras.layers.BatchNormalization()
        # === Fourth path for the 3x3 max-pool layer ===
        self.avg_pool2d = tf.keras.layers.AveragePooling2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf4 = tf.keras.layers.BatchNormalization()
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)
        bn_1_nf1 = self.bn_1_nf1(conv2d_1_nf1, training=training)

        # === Second path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        bn_1_nf2_a = self.bn_1_nf2_a(conv2d_1_nf2_a, training=training)
        conv2d_3_nf2_b = self.conv2d_3_nf2_b(bn_1_nf2_a)
        bn_3_nf2_b = self.bn_3_nf2_b(conv2d_3_nf2_b, training=training)

        # === Third path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        bn_1_nf3_a = self.bn_1_nf3_a(conv2d_1_nf3_a, training=training)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(bn_1_nf3_a)
        bn_3_nf3_b_i = self.bn_3_nf3_b_i(conv2d_3_nf3_b_i, training=training)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(bn_3_nf3_b_i)
        bn_3_nf3_b_ii = self.bn_3_nf3_b_ii(conv2d_3_nf3_b_ii, training=training)

        # === Fourth path for the 3x3 max-pool layer ===
        avg_pool2d = self.avg_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(avg_pool2d)
        bn_1_nf4 = self.bn_1_nf4(conv2d_1_nf4, training=training)

        # === Concatenation ===
        concatenation = self.concatenation([bn_1_nf1, 
                                            bn_3_nf2_b, 
                                            bn_3_nf3_b_ii, 
                                            bn_1_nf4])

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

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(224, 224, 3))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))

class InceptionModuleWithAsymmetricFactorization(tf.keras.layers.Layer):
    '''
    Inception module with asymmetric factorization
    '''
    def __init__(self, nf1, nf2_a, nf2_b, nf3_a, nf3_b, nf4, **kwargs):
        super(InceptionModuleWithAsymmetricFactorization, self).__init__(**kwargs)
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
        self.bn_1_nf1 = tf.keras.layers.BatchNormalization()
        # === Second path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf2_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf2_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                       (1, 7),
                                                       padding='same',
                                                       activation='relu')
        self.bn_3_nf2_b_i = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf2_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                        (7, 1),
                                                        padding='same',
                                                        activation='relu')
        self.bn_3_nf2_b_ii = tf.keras.layers.BatchNormalization()
        # === Third path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf3_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                       (7, 1),
                                                       padding='same',
                                                       activation='relu')
        self.bn_3_nf3_b_i = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                        (1, 7),
                                                        padding='same',
                                                        activation='relu')
        self.bn_3_nf3_b_ii = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_iii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                         (7, 1),
                                                         padding='same',
                                                         activation='relu')
        self.bn_3_nf3_b_iii = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_iv = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                        (1, 7),
                                                        padding='same',
                                                        activation='relu')
        self.bn_3_nf3_b_iv = tf.keras.layers.BatchNormalization()
        # === Fourth path for the 3x3 max-pool layer ===
        self.avg_pool2d = tf.keras.layers.AveragePooling2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        self.bn_1_nf4 = tf.keras.layers.BatchNormalization()

        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)
        bn_1_nf1 = self.bn_1_nf1(conv2d_1_nf1, training=training)

        # === Second path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        bn_1_nf2_a = self.bn_1_nf2_a(conv2d_1_nf2_a, training=training)
        conv2d_3_nf2_b_i = self.conv2d_3_nf2_b_i(bn_1_nf2_a)
        bn_3_nf2_b_i = self.bn_3_nf2_b_i(conv2d_3_nf2_b_i, training=training)
        conv2d_3_nf2_b_ii = self.conv2d_3_nf2_b_ii(bn_3_nf2_b_i)
        bn_3_nf2_b_ii = self.bn_3_nf2_b_ii(conv2d_3_nf2_b_ii, training=training)

        # === Third path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        bn_1_nf3_a = self.bn_1_nf3_a(conv2d_1_nf3_a, training=training)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(bn_1_nf3_a)
        bn_3_nf3_b_i = self.bn_3_nf3_b_i(conv2d_3_nf3_b_i, training=training)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(bn_3_nf3_b_i)
        bn_3_nf3_b_ii = self.bn_3_nf3_b_ii(conv2d_3_nf3_b_ii, training=training)
        conv2d_3_nf3_b_iii = self.conv2d_3_nf3_b_iii(bn_3_nf3_b_ii)
        bn_3_nf3_b_iii = self.bn_3_nf3_b_iii(conv2d_3_nf3_b_iii, training=training)
        conv2d_3_nf3_b_iv = self.conv2d_3_nf3_b_iv(bn_3_nf3_b_iii)
        bn_3_nf3_b_iv = self.bn_3_nf3_b_iv(conv2d_3_nf3_b_iv, training=training)

        # === Fourth path for the 3x3 max-pool layer ===
        avg_pool2d = self.avg_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(avg_pool2d)
        bn_1_nf4 = self.bn_1_nf4(conv2d_1_nf4, training=training)

        # === Concatenation ===
        concatenation = self.concatenation([bn_1_nf1, 
                                            bn_3_nf2_b_ii, 
                                            bn_3_nf3_b_iv, 
                                            bn_1_nf4])

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

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(224, 224, 3))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))

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
        self.bn_1_nf1 = tf.keras.layers.BatchNormalization()
        # === Second path for the 3x3 convolutional layer ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf2_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf2_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                       (1, 3),
                                                       padding='same',
                                                       activation='relu')
        self.bn_3_nf2_b_i = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf2_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_2_b,
                                                        (3, 1),
                                                        padding='same',
                                                        activation='relu')
        self.bn_3_nf2_b_ii = tf.keras.layers.BatchNormalization()
        # === Third path for the 5x5 convolutional layer ===
        self.conv2d_1_nf3_a = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf3_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_i = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                       (3, 3),
                                                       padding='same',
                                                       activation='relu')
        self.bn_3_nf3_b_i = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_ii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                        (1, 3),
                                                        padding='same',
                                                        activation='relu')
        self.bn_3_nf3_b_ii = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf3_b_iii = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_3_b,
                                                         (3, 1),
                                                         padding='same',
                                                         activation='relu')
        self.bn_3_nf3_b_iii = tf.keras.layers.BatchNormalization()
        # === Fourth path for the 3x3 max-pool layer ===
        self.avg_pool2d = tf.keras.layers.AveragePooling2D((3, 3), 
                                                    strides=(1, 1), 
                                                    padding='same')
        self.conv2d_1_nf4 = tf.keras.layers.Conv2D(self.n_filters_of_conv_layer_4,
                                                   (1, 1),
                                                   padding='same',
                                                   activation='relu')
        self.bn_1_nf4 = tf.keras.layers.BatchNormalization()

        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path for the 1x1 convolutional layer ===
        conv2d_1_nf1 = self.conv2d_1_nf1(input_tensor)
        bn_1_nf1 = self.bn_1_nf1(conv2d_1_nf1, training=training)

        # === Second path for the 3x3 convolutional layer ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        bn_1_nf2_a = self.bn_1_nf2_a(conv2d_1_nf2_a, training=training)
        conv2d_3_nf2_b_i = self.conv2d_3_nf2_b_i(bn_1_nf2_a)
        bn_3_nf2_b_i = self.bn_3_nf2_b_i(conv2d_3_nf2_b_i, training=training)
        conv2d_3_nf2_b_ii = self.conv2d_3_nf2_b_ii(bn_1_nf2_a)
        bn_3_nf2_b_ii = self.bn_3_nf2_b_ii(conv2d_3_nf2_b_ii, training=training)

        # === Third path for the 5x5 convolutional layer ===
        conv2d_1_nf3_a = self.conv2d_1_nf3_a(input_tensor)
        bn_1_nf3_a = self.bn_1_nf3_a(conv2d_1_nf3_a, training=training)
        conv2d_3_nf3_b_i = self.conv2d_3_nf3_b_i(bn_1_nf3_a)
        bn_3_nf3_b_i = self.bn_3_nf3_b_i(conv2d_3_nf3_b_i, training=training)
        conv2d_3_nf3_b_ii = self.conv2d_3_nf3_b_ii(bn_3_nf3_b_i)
        bn_3_nf3_b_ii = self.bn_3_nf3_b_ii(conv2d_3_nf3_b_ii, training=training)
        conv2d_3_nf3_b_iii = self.conv2d_3_nf3_b_iii(bn_3_nf3_b_i)
        bn_3_nf3_b_iii = self.bn_3_nf3_b_iii(conv2d_3_nf3_b_iii, training=training)

        # === Fourth path for the 3x3 max-pool layer ===
        avg_pool2d = self.avg_pool2d(input_tensor)
        conv2d_1_nf4 = self.conv2d_1_nf4(avg_pool2d)
        bn_1_nf4 = self.bn_1_nf4(conv2d_1_nf4, training=training)

        # === Concatenation ===
        concatenation = self.concatenation([bn_1_nf1, 
                                            bn_3_nf2_b_i, 
                                            bn_3_nf2_b_ii, 
                                            bn_3_nf3_b_ii, 
                                            bn_3_nf3_b_iii, 
                                            bn_1_nf4])

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

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(224, 224, 3))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))

class AuxiliaryClassifier(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AuxiliaryClassifier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.average_pooling = tf.keras.layers.AveragePooling2D(pool_size=(5, 5),
                                                                strides=(3, 3),
                                                                padding='valid')
        self.conv2d_5_a = tf.keras.layers.Conv2D(128,
                                                 (1, 1),
                                                 padding='same')
        self.bn_5_a = tf.keras.layers.BatchNormalization()
        self.conv2d_5_b = tf.keras.layers.Conv2D(768,
                                                 (5, 5),
                                                 padding='valid',
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(0.01))
        self.bn_5_b = tf.keras.layers.BatchNormalization()
        self.conv2d_5_c = tf.keras.layers.Conv2D(3,
                                                 (1, 1),
                                                 activation=None,
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(0.001))
        self.bn_5_c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        print('[*] input_tensor shape: ', input_tensor.shape)
        average_pooling = self.average_pooling(input_tensor)
        print('[*] average_pooling shape: ', average_pooling.shape)
        conv2d_5_a = self.conv2d_5_a(average_pooling)
        bn_5_a = self.bn_5_a(conv2d_5_a, training=training)
        print('[*] bn_5_a shape: ', bn_5_a.shape)
        conv2d_5_b = self.conv2d_5_b(bn_5_a)
        bn_5_b = self.bn_5_b(conv2d_5_b, training=training)
        print('[*] bn_5_b shape: ', bn_5_b.shape)
        conv2d_5_c = self.conv2d_5_c(bn_5_b)
        print('[*] conv2d_5_c shape: ', conv2d_5_c.shape)
        bn_5_c = self.bn_5_c(conv2d_5_c, training=training)
        print('[*] bn_5_c shape: ', bn_5_c.shape)
        squeeze = tf.squeeze(bn_5_c, [1, 2])

        return squeeze

    def get_config(self):
        config = super().get_config().copy()
        return config

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(17, 17, 768))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))

class GridSizeReduction(tf.keras.layers.Layer):
    def __init__(self, nf1_a, nf1_b, nf2_a, nf2_b, **kwargs):
        super(GridSizeReduction, self).__init__(**kwargs)
        self.n_filters_of_layer_1_a = nf1_a
        self.n_filters_of_layer_1_b = nf1_b
        self.n_filters_of_layer_2_a = nf2_a
        self.n_filters_of_layer_2_b = nf2_b

    def build(self, input_shape):
        # === First path ===
        self.conv2d_1_nf1_a = tf.keras.layers.Conv2D(self.n_filters_of_layer_1_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf1_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf1_b = tf.keras.layers.Conv2D(self.n_filters_of_layer_1_b,
                                                     (3, 3),
                                                     padding='same',
                                                     activation='relu')
        self.bn_3_nf1_b = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf1_c = tf.keras.layers.Conv2D(self.n_filters_of_layer_1_b,
                                                     (3, 3),
                                                     strides=(2, 2),
                                                     padding='valid',
                                                     activation='relu')
        self.bn_3_nf1_c = tf.keras.layers.BatchNormalization()

        # === Second path ===
        self.conv2d_1_nf2_a = tf.keras.layers.Conv2D(self.n_filters_of_layer_2_a,
                                                     (1, 1),
                                                     padding='same',
                                                     activation='relu')
        self.bn_1_nf2_a = tf.keras.layers.BatchNormalization()
        self.conv2d_3_nf2_b = tf.keras.layers.Conv2D(self.n_filters_of_layer_2_b,
                                                     (3, 3),
                                                     strides=(2, 2),
                                                     padding='valid',
                                                     activation='relu')
        self.bn_3_nf2_b = tf.keras.layers.BatchNormalization()

        # === Third path ===
        self.max_pool2d = tf.keras.layers.MaxPool2D((3, 3),
                                                     strides=(2, 2),
                                                     padding='valid')

        # === Concatenation ===
        self.concatenation = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_tensor, training=False):
        # === First path ===
        conv2d_1_nf1_a = self.conv2d_1_nf1_a(input_tensor)
        bn_1_nf1_a = self.bn_1_nf1_a(conv2d_1_nf1_a)
        conv2d_3_nf1_b = self.conv2d_3_nf1_b(bn_1_nf1_a)
        bn_3_nf1_b = self.bn_3_nf1_b(conv2d_3_nf1_b)
        conv2d_3_nf1_c = self.conv2d_3_nf1_c(bn_3_nf1_b)
        bn_3_nf1_c = self.bn_3_nf1_c(conv2d_3_nf1_c)

        # === Second path ===
        conv2d_1_nf2_a = self.conv2d_1_nf2_a(input_tensor)
        bn_1_nf2_a = self.bn_1_nf2_a(conv2d_1_nf2_a)
        conv2d_3_nf2_b = self.conv2d_3_nf2_b(bn_1_nf2_a)
        bn_3_nf2_b = self.bn_3_nf2_b(conv2d_3_nf2_b)

        # === Third path ===
        max_pool2d = self.max_pool2d(input_tensor)

        # === Concatenation ===
        concatenation = self.concatenation([bn_3_nf1_c,
                                            bn_3_nf2_b,
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

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(35, 35, 320))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))