"""
ConvMixer.py - TensorFlow implementation of
ICLR 2022 submission "Patches Are All You Need?"

Dan Mezhiborsky - @dmezh
Theo Jaquenoud - @thjaquenoud

See:
https://github.com/dmezh/convmixer-tf
https://github.com/tmp-iclr/convmixer

Our final layer uses softmax activation.
"""

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras


class Residual(keras.layers.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x


def GELU():
    return keras.layers.Activation(tf.keras.activations.gelu)


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=10):
    return keras.Sequential(
        [
            keras.layers.Conv2D(
                dim,
                kernel_size=(patch_size, patch_size),
                strides=(patch_size, patch_size),
                input_shape=(32, 32, 3),
            ),
            GELU(),
            keras.layers.BatchNormalization(),
            *[
                keras.Sequential(
                    [
                        Residual(
                            keras.Sequential(
                                [
                                    keras.layers.Conv2D(
                                        dim,
                                        kernel_size=(kernel_size, kernel_size),
                                        groups=dim,
                                        padding="same",
                                    ),
                                    GELU(),
                                    keras.layers.BatchNormalization(),
                                ]
                            )
                        ),
                        keras.layers.Conv2D(dim, kernel_size=(1, 1)),
                        GELU(),
                        keras.layers.BatchNormalization(),
                    ]
                )
                for i in range(depth)
            ],
            tfa.layers.AdaptiveAveragePooling2D((1, 1)),
            keras.layers.Flatten(),
            keras.layers.Activation(tf.keras.activations.linear),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
