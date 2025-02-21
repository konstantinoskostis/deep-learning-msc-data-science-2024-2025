"""A module for providing tuning and training capabilities for Convolutional Neural Networks"""

import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Layer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import keras_tuner as kt
import matplotlib.pyplot as plt

from evaluation_utils import (Metrics, EvaluationReport)
from constants import SEED


class Conv2DBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', padding='same',
                 max_pooling_size=(2, 2), strides=1, dropout_rate=0.1):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.max_pooling_size = max_pooling_size
        self.strides = strides
        self.dropout_rate = dropout_rate

        self.convolution = Conv2D(filters=filters, kernel_size=self.kernel_size,
                                  activation=self.activation, padding=self.padding)

        if self.max_pooling is not None:
            self.pooling = MaxPooling2D(
                pool_size=self.max_pooling_size, strides=self.strides)

        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs):
        x = self.convolution(inputs)

        if self.max_pooling_size is not None:
            x = self.pooling(x)

        if self.dropout_rate is not None:
            x = self.dropout(x)

        return x


class CNNTuner:
    def __init__(self,
                 train_dataset, validation_dataset, tuner_directory, project_name,
                 train_size=0.5, seed=SEED, max_conv2d_blocks=3, use_dropout=True):
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.train_size = train_size
        self.seed = seed
        self.max_conv2d_blocks = max_conv2d_blocks
        self.use_dropout = use_dropout

        X_rem, X_train, y_rem, y_train = train_test_split(
            train_dataset[0].copy(), train_dataset[1].copy(),
            stratify=train_dataset[1].copy(),
            test_size=self.train_size,
            random_state=self.seed)

        self.val_X, self.val_y = validation_dataset
        self.train_X = X_train
        self.train_y = y_train

    def tune(self, max_trials=10, epochs=30, batch_size=128, patience=5):
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

        # Add a simple convolutional layer with input dimensions
        kernel = hp.choice(name='kernel', values=[3, 5, 7])
        conv = Conv2D(
            filters=hp.Int('conv_filters', min_value=32,
                           max_value=128, step=32),
            kernel_size=(kernel, kernel),
            activation=hp.Choice("conv_activation", ["relu", "tanh"]),
            padding='same',
            input_shape=self.train_X.shape[1:])

        model.add(conv)

        # Stack a number of custom Conv2DBlock objects
        num_layers = hp.Int(name='num_layers', min_value=1,
                            max_value=self.max_conv2d_blocks)

        for i in range(num_layers):
            layer_kernel = hp.choice(
                name='layer_{}_kernel'.format(i), values=[3, 5, 7])

            layer_pool_size = hp.choice(
                name='layer_{}_pool_size'.format(i), values=[2, 3])
            layer_pool_strides = hp.choice(
                name='layer_{}_pool_strides'.format(i), values=[1, 2])

            layer_dropout_rate = 0.0

            if self.use_dropout:
                layer_dropout_rate = hp.Float(
                    'layer_{}_dropout'.format(i),
                    0.1, 0.5, step=0.05)

            conv_block = Conv2DBlock(
                filters=hp.Int('conv2dblock_{}_filters'.format(i),
                               min_value=32, max_value=128, step=32),
                kernel_size=(layer_kernel, layer_kernel),
                activation=hp.Choice(
                    name="conv2dblock_{}_activation", values=["relu", "tanh"]),
                padding='same',
                max_pooling_size=(layer_pool_size, layer_pool_size),
                strides=layer_pool_strides,
                dropout_rate=layer_dropout_rate
            )
            model.add(conv_block)

        # Add a flattening Layer
        model.add(Flatten())

        # Add an MLP (for classification) layer
        model.add(Dense(
            units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
            activation='relu')
        )
        model.add(Dropout(hp.Float('dense_dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(self.train_y.shape[1], activation='softmax'))

        # Optimizer tuning
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=hp_learning_rate),
                      metrics=[CategoricalAccuracy()])

        return model
