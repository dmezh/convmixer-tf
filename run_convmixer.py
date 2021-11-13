#!/usr/bin/python3

"""
ConvMixer-tf CIFAR10 Testbench

Dan Mezhiborsky - @dmezh
Theo Jaquenoud - @thjaquenoud

See:
https://github.com/dmezh/convmixer-tf
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from imgaug import augmenters as iaa

from load_cifar_10 import load_cifar_10_data

from ConvMixer import ConvMixer
from OneCycleLR import OneCycleLRScheduler

AUTO = tf.data.AUTOTUNE
rand_aug = iaa.RandAugment(n=3, m=7)

MODEL_WIDTH = 128
MODEL_DEPTH = 4
KERNEL_SIZE = 8
PATCH_SIZE = 1

CIFAR_10_DIR = "cifar-10-python/cifar-10-batches-py"
BATCH_SIZE = 32
NUM_EPOCHS = 200

MAX_LR = 0.01

INITIAL_EPOCH = 0
LOAD_SAVED_MODEL = False
SAVED_MODEL_PATH = (
    "aug_onecyclecmx_0.01_nodropout_noreg_200/cifar_10_model_128x12_epoch_125-0.92/"
)

SAVE_PERIOD = 5

EXPERIMENT_NAME = (
    f"aug_onecyclecmx_lr-{MAX_LR}_bs-{BATCH_SIZE}"
    f"_nodropout_noreg_{NUM_EPOCHS}_saved-{LOAD_SAVED_MODEL}"
)

# -----------------------------------------------------------------------------


def augment(images):
    # Input to `augment()` is a TensorFlow tensor, which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())


def main():
    (
        images,
        train_filenames,
        labels,
        t_images,
        test_filenames,
        t_labels,
        label_names,
    ) = load_cifar_10_data(CIFAR_10_DIR)

    images_conv = (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
        .map(
            lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y),
            num_parallel_calls=AUTO,
        )
        .prefetch(AUTO)
    )

    t_images_conv = (
        tf.data.Dataset.from_tensor_slices((t_images, t_labels))
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    if LOAD_SAVED_MODEL:
        model = keras.models.load_model(SAVED_MODEL_PATH)
    else:
        model = ConvMixer(
            MODEL_WIDTH,
            MODEL_DEPTH,
            kernel_size=KERNEL_SIZE,
            patch_size=PATCH_SIZE,
            n_classes=10,
        )
        model.compile(
            optimizer=tfa.optimizers.AdamW(weight_decay=0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.build()
        model.summary()

    filename_prefix = "./{}/cifar_10_model_{}x{}".format(
        EXPERIMENT_NAME, MODEL_WIDTH, MODEL_DEPTH
    )
    filename = filename_prefix + "_epoch_{epoch:02d}-{val_accuracy:.2f}"

    checkpt = tf.keras.callbacks.ModelCheckpoint(filename, period=SAVE_PERIOD)

    lrsched = OneCycleLRScheduler(NUM_EPOCHS, MAX_LR, images.shape[0] / BATCH_SIZE)

    history = model.fit(
        images_conv,
        initial_epoch=INITIAL_EPOCH,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=t_images_conv,
        callbacks=[checkpt, lrsched],
    )

    np.save(f"./{EXPERIMENT_NAME}/history.npy", history.history)


if __name__ == "__main__":
    main()
