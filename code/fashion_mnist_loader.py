"""A module for providing data loading capabilities for Fashion MNIST dataset"""

import gzip
import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from constants import SEED


class FashionMNISTLoader:
    """A data loader class for the Fashion MNIST dataset"""

    def __init__(self, path, kind, seed=SEED):
        """Initializes the data loader

        Args:
            path: A string path/directory from where to read the data
            kind: The type of dataset to read ('train' for training and 't10k' for testing)
            seed: A random seed for reproducibility
        """
        self.path = path
        self.kind = kind
        self.seed = seed

    def load_data(self, validation_size=0.0, normalize=True, as_categorical=True, flatten_shape=True):
        """Loads the data

        Args:
            validation_size (float, optional): The percentage to use for validation data
            normalize (boolean, optional): Whether or not to normalize the image data
            as_categorical (boolean, optional): Whether or not to convert the labels to one-hot vectors
            flatten_shape (boolean, optional): Whether or not to reshape an image as a vector or not

        Returns:
            Tuple: Either in format (X, y) when validation_size is 0 or
                   in format ((X, y), (val_X, val_y)) when validation_size is > 0
        """
        loaded_data, loaded_labels = FashionMnistLoader.load(
            self.path,
            kind=self.kind, normalize=normalize,
            as_categorical=as_categorical, flatten_shape=flatten_shape)

        if validation_size > 0.0:
            X, val_X, y, val_y = train_test_split(loaded_data, loaded_labels,
                                                  stratify=loaded_labels,
                                                  test_size=validation_size,
                                                  random_state=self.seed)
            return ((X, y), (val_X, val_y))
        else:
            return (loaded_data, loaded_labels)

    @staticmethod
    def load(path, kind='train', normalize=True, as_categorical=True, flatten_shape=True):
        """A static method that loads the data and labels of fashion MNIST dataset

        Args:
            path: A string path/directory from where to read the data
            kind: The type of dataset to read ('train' for training and 't10k' for testing)
            normalize (boolean, optional): Whether or not to normalize the image data
            as_categorical (boolean, optional): Whether or not to convert the labels to one-hot vectors
            flatten_shape (boolean, optional): Whether or not to reshape an image as a vector or not
        """
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

            if as_categorical is True:
                unique = np.unique(labels)
                labels = to_categorical(labels, num_classes=len(unique))

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)

            if flatten_shape is True:
                images = images.reshape((len(labels), 784))
            else:
                images = images.reshape((len(labels), 28, 28, 1))

            if normalize is True:
                images = images / 255.0

        return images, labels
