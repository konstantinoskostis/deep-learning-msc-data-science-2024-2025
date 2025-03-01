"""This module contains code regarding the custom architecture of the neural net for the ESC-50 dataset"""

import os

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Layer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import keras_tuner as kt
import matplotlib.pyplot as plt

from evaluation_utils import (Metrics, EvaluationReport)
from constants import SEED
from cnn import Conv2DBlock

# Set tensorflow random seed for reproducibility
tf.random.set_seed(SEED)


class ESC50CNNTuner:
    def __init__(self,
                 train_dataset, validation_dataset, tuner_directory, project_name,
                 train_size=0.5, seed=SEED, max_conv2d_blocks=3, output_dims=128,
                 use_dropout=True):
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.train_size = train_size
        self.seed = seed
        self.max_conv2d_blocks = max_conv2d_blocks
        self.output_dims = output_dims
        self.use_dropout = use_dropout

        X_rem, X_train, y_rem, y_train = train_test_split(
            train_dataset[0].copy(), train_dataset[1].copy(),
            stratify=train_dataset[1].copy(),
            test_size=self.train_size,
            random_state=self.seed)

        self.val_X, self.val_y = validation_dataset
        self.train_X = X_train
        self.train_y = y_train

    def tune(self, max_trials=10, epochs=30, batch_size=32, patience=5):
        tuner = kt.RandomSearch(self.build_model,
                                objective=kt.Objective(
                                    "val_loss", direction="min"),
                                max_trials=max_trials,
                                seed=self.seed,
                                directory=self.tuner_directory,
                                project_name=self.project_name)

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        tuner.search(self.train_X, self.train_y,
                     validation_data=(self.val_X, self.val_y),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[early_stopping])

        return tuner

    def build_model(self, hp):
        model = Sequential()
        model.add(Input(shape=self.train_X.shape[1:]))

        # Stack a number of custom Conv2DBlock objects
        num_layers = hp.Int(name='num_layers', min_value=1,
                            max_value=self.max_conv2d_blocks)

        for i in range(num_layers):
            layer_filters = hp.Choice(
                'conv2dblock_{}_filters'.format(i), [32, 64, 128])

            layer_dropout_rate = 0.0

            if self.use_dropout:
                layer_dropout_rate = hp.Float(
                    'layer_{}_dropout'.format(i),
                    0.1, 0.5, step=0.05)

            conv_block = Conv2DBlock(
                filters=layer_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same',
                max_pooling_size=(2, 2),
                dropout_rate=layer_dropout_rate
            )
            model.add(conv_block)

        # Add a flattening Layer and reduce to 128-dimensional feature vector
        model.add(Flatten())
        model.add(Dense(self.output_dims, activation='relu'))

        # Add a Dense classification layer
        model.add(Dense(self.train_y.shape[1], activation='softmax'))

        # Optimizer tuning
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=hp_learning_rate),
                      metrics=[CategoricalAccuracy()])

        return model
