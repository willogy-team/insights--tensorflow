import numpy as np
import tensorflow as tf


class VGG16Net(tf.keras.Model):
    def __init__(self, num_classes=3):
        super(VGG16Net, self).__init__()
        # self.block_1 = VGGBlock(conv_layers=2, filters=64)
        # self.block_2 = VGGBlock(conv_layers=2, filters=128)
        # self.block_3 = VGGBlock(conv_layers=3, filters=256)
        # self.block_4 = VGGBlock(conv_layers=3, filters=512)
        # self.block_5 = VGGBlock(conv_layers=3, filters=512)
        # self.flatten = tf.keras.layers.Flatten(input_shape=(7, 7, 512))
        # self.dense_1 = tf.keras.layers.Dense(4096, activation='relu')
        # self.dense_2 = tf.keras.layers.Dense(4096, activation='relu')
        # self.dense_3 = tf.keras.layers.Dense(4096, activation='relu')
        # self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

        self.block_1 = VGGBlock(conv_layers=1, filters=64)
        self.block_2 = VGGBlock(conv_layers=1, filters=128)
        self.block_3 = VGGBlock(conv_layers=1, filters=256)
        self.block_4 = VGGBlock(conv_layers=1, filters=512)
        self.block_5 = VGGBlock(conv_layers=2, filters=512)
        self.flatten = tf.keras.layers.Flatten(input_shape=(7, 7, 512))
        self.dense_1 = tf.keras.layers.Dense(496, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(496, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(496, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        print('[+] inputs shape: ', inputs.shape)
        x = self.block_1(inputs)
        print('[+] self.block_1 shape: ', x.shape)
        x = self.block_2(x)
        print('[+] self.block_2 shape: ', x.shape)
        x = self.block_3(x)
        print('[+] self.block_3 shape: ', x.shape)
        x = self.block_4(x)
        print('[+] self.block_4 shape: ', x.shape)
        x = self.block_5(x)
        print('[+] self.block_5 shape: ', x.shape)
        x = self.flatten(x)
        print('[+] self.flatten shape: ', x.shape)
        x = self.dense_1(x)
        print('[+] self.dense_1 shape: ', x.shape)
        x = self.dense_2(x)
        print('[+] self.dense_2 shape: ', x.shape)
        x = self.dense_3(x)
        print('[+] self.dense_3 shape: ', x.shape)
        x = self.classifier(x)
        # print('[+] self.classifier shape: ', tf.shape(x))
        print('[+] self.classifier shape: ', x.shape)
        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class VGGBlock(tf.keras.Model):
    def __init__(self, conv_layers=2, kernel_size=3, filters=64):
        super(VGGBlock, self).__init__(name='')
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.filters = filters

        self.layer_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for i in range(self.conv_layers):
            # --- Option 1 ---
            left_cls = ''.join(["self.conv2d", "_", str(self.kernel_size), "_", str(self.filters), "_", str(self.layer_id[i])])
            right_cls = ''.join(["tf.keras.layers.Conv2D(", str(self.filters), ", ", "(", str(self.kernel_size), ", ", str(self.kernel_size), ")", ", ",
                                "activation='relu'", ", ", "padding='same'", ")"])
            assignation = ''.join([left_cls, '=', right_cls])
            # print('[**] assignation: ', assignation)
            exec(assignation)
        
        # print('[**] ', self.conv2d_3_64_a)
        # print('[**] ', self.conv2d_3_64_b)
        self.max_pool2d = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')

    # def compute_output_shape(self, input_shape):
    #     # return super().compute_output_shape(input_shape)
    #     return (input_shape[0], input_shape[1], input_shape[2])

    def call(self, input_tensor, training=False):
        # print('[**] ', self.conv2d_3_64_a)
        # print('[**] ', self.conv2d_3_64_b)
        for i in range(self.conv_layers):
            # layer = ''.join(["self.conv2d", "_", str(self.kernel_size), "_", str(self.filters), "_", str(self.layer_id[i])])
            # print('[**] layer: ', eval(layer))
            if i == 0:
                layer = ''.join(["self.conv2d", "_", str(self.kernel_size), "_", str(self.filters), "_", str(self.layer_id[i])])
                # print('[**] layer: ', eval(layer))
                x = eval(layer)(input_tensor)
            else:
                layer = ''.join(["self.conv2d", "_", str(self.kernel_size), "_", str(self.filters), "_", str(self.layer_id[i])])
                # print('[**] layer: ', eval(layer))
                x = eval(layer)(x)
        x = self.max_pool2d(x)
        return x

# class VGGBlock(tf.keras.Model):
#     def __init__(self, conv_layers=2, kernel_size=3, filters=64):
#         super(VGGBlock, self).__init__(name='')
#         self.conv_layers = conv_layers
#         self.kernel_size = kernel_size
#         self.filters = filters

#         self.layer_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
#         for i in range(self.conv_layers):
#             # --- Option 1 ---
#             left_cls = ''.join(["self.conv2d", "_", str(self.kernel_size), "_", str(self.filters), "_", str(self.layer_id[i])])
#             # right_cls = ''.join(["tf.keras.layers.Conv2D(", filters, ", ", "(", kernel_size, ", ", kernel_size, ")", ", ",
#             #                     "activation='relu'", ")"])
#             # assignation = ''.join([left_cls, '=', right_cls])
#             # exec(assignation)
#             # --- Option 2 ---
#             globals()[left_cls] = tf.keras.layers.Conv2D(self.filters, self.kernel_size, activation='relu', padding='same')
        
#         print('[**] ', globals()["self.conv2d_3_64_a"])
#         print('[**] ', globals()["self.conv2d_3_64_b"])
#         self.max_pool2d = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')

#     def call(self, input_tensor, training=False):
#         print('[***] ', globals()["self.conv2d_3_64_a"])
#         print('[***] ', globals()["self.conv2d_3_64_b"])
#         for i in range(self.conv_layers):
#             layer = ''.join(["self.conv2d", "_", str(self.kernel_size), "_", str(self.filters), "_", str(self.layer_id[i])])
#             if i == 0:
#                 x = globals()[layer](input_tensor)
#             else:
#                 x = globals()[layer](x)
#         x = self.max_pool2d(x)
#         return x