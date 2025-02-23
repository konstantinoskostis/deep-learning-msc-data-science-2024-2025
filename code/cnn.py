"""A module for providing tuning and training capabilities for Convolutional Neural Networks"""

import os

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
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

# Set tensorflow random seed for reproducibility
tf.random.set_seed(SEED)


class Conv2DBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', padding='same',
                 max_pooling_size=(2, 2), dropout_rate=0.1):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.max_pooling_size = max_pooling_size
        self.dropout_rate = dropout_rate

        self.convolution = Conv2D(filters=filters,
                                  kernel_size=self.kernel_size, strides=self.strides,
                                  activation=self.activation, padding=self.padding)

        if self.max_pooling_size is not None:
            self.pooling = MaxPooling2D(pool_size=self.max_pooling_size)

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
        kernel = hp.Choice(name='kernel_size', values=[3])
        kernel_stride = hp.Choice(name='kernel_stride', values=[1])
        conv = Conv2D(
            filters=hp.Int('conv_filters', min_value=32,
                           max_value=128, step=32),
            kernel_size=(kernel, kernel),
            strides=(kernel_stride, kernel_stride),
            activation=hp.Choice("conv_activation", ["relu", "tanh"]),
            padding='same',
            input_shape=self.train_X.shape[1:])

        model.add(conv)

        # Stack a number of custom Conv2DBlock objects
        num_layers = hp.Int(name='num_layers', min_value=1,
                            max_value=self.max_conv2d_blocks)

        for i in range(num_layers):
            layer_kernel = hp.Choice(
                name='layer_{}_kernel_size'.format(i), values=[3])
            layer_kernel_stride = hp.Choice(
                name='layer_{}_kernel_stride'.format(i), values=[2])
            layer_pool_size = hp.Choice(
                name='layer_{}_pool_size'.format(i), values=[2])

            layer_dropout_rate = 0.0

            if self.use_dropout:
                layer_dropout_rate = hp.Float(
                    'layer_{}_dropout'.format(i),
                    0.1, 0.5, step=0.05)

            conv_block = Conv2DBlock(
                filters=hp.Int('conv2dblock_{}_filters'.format(i),
                               min_value=32, max_value=128, step=32),
                kernel_size=(layer_kernel, layer_kernel),
                strides=(layer_kernel_stride, layer_kernel_stride),
                activation=hp.Choice(
                    name="conv2dblock_{}_activation".format(i),
                    values=["relu", "tanh"]),
                padding='same',
                max_pooling_size=(layer_pool_size, layer_pool_size),
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


class CNN:

    def __init__(self, models_dir, weights_name, class_ids, class_labels, seed=SEED):
        self.models_dir = models_dir
        self.weights_name = weights_name
        self.weights_path = os.path.join(self.models_dir, self.weights_name)
        self.seed = seed
        self.class_ids = class_ids
        self.class_labels = class_labels

        self.model = None
        self.history = None

    def fit(self,
            train_dataset, validation_dataset, hyperparams,
            batch_size=128, epochs=100, patience=10):
        X_train, y_train = train_dataset
        X_val, y_val = validation_dataset

        self.build_model(hyperparams, X_train.shape[-1], y_train.shape[-1])

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        checkpoint = ModelCheckpoint(self.weights_path,
                                     monitor='val_f1',
                                     mode='max', verbose=2,
                                     save_best_only=True,
                                     save_weights_only=True)

        early_stopping = EarlyStopping(patience=patience, verbose=2,
                                       restore_best_weights=True,
                                       monitor='val_f1', mode='max')

        self.history = self.model.fit(X_train, y_train,
                                      validation_data=(X_val, y_val),
                                      batch_size=batch_size, epochs=epochs, shuffle=True,
                                      callbacks=[
                                          Metrics(valid_data=(X_val, y_val)),
                                          checkpoint,
                                          early_stopping
                                      ])

    def classification_report(self, dataset):
        """ Create a classification report for a given dataset."""
        data_X, data_y = dataset
        y_true = np.argmax(data_y, -1)
        y_probabilities = self.model.predict(data_X)
        y_predicted = np.argmax(y_probabilities.copy(), -1)

        return EvaluationReport.classification_report(
            y_true, y_probabilities, y_predicted,
            self.class_ids, self.class_labels)

    def build_model(self, hyperparams, input_dim, output_dim):
        """Builds a configurable CNN model based on given hyperparameters."""

        self.model = Sequential()

        # Add a simple convolutional layer with input dimensions
        kernel = hyperparams.get('kernel_size', 3)
        kernel_stride = hyperparams.get('kernel_stride', 1)
        conv = Conv2D(
            filters=hyperparams.get('conv_filters', 32),
            kernel_size=(kernel, kernel),
            strides=(kernel_stride, kernel_stride),
            activation=hyperparams.get('conv_activation', 'relu'),
            padding='same',
            input_shape=input_dim)

        self.model.add(conv)

        # Stack a number of custom Conv2DBlock objects
        num_layers = hyperparams.get('num_layers', 1)

        for i in range(num_layers):
            layer_kernel = hyperparams.get('layer_{}_kernel_size'.format(i), 3)
            layer_kernel_stride = hyperparams.get(
                'layer_{}_kernel_stride'.format(i), 2)
            layer_pool_size = hyperparams.get(
                'layer_{}_pool_size'.format(i), 2)

            layer_dropout_rate = hyperparams.get(
                'layer_{}_dropout'.format(i), 0.0)

            conv_block = Conv2DBlock(
                filters=hyperparams.get(
                    'conv2dblock_{}_filters'.format(i), 32),
                kernel_size=(layer_kernel, layer_kernel),
                strides=(layer_kernel_stride, layer_kernel_stride),
                activation=hyperparams.get(
                    'conv2dblock_{}_activation'.format(i), 'relu'),
                padding='same',
                max_pooling_size=(layer_pool_size, layer_pool_size),
                dropout_rate=layer_dropout_rate
            )
            self.model.add(conv_block)

        # Add a flattening Layer
        self.model.add(Flatten())

        # Add an MLP (for classification) layer
        self.model.add(Dense(
            units=hyperparams.get('dense_units', 64),
            activation='relu')
        )
        self.model.add(
            Dropout(hyperparams.get('dense_dropout', 0.1))
        )
        self.model.add(Dense(output_dim, activation='softmax'))

        # Optimizer tuning
        learning_rate = hyperparams.get('learning_rate', 1e-3)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=learning_rate),
                           metrics=[CategoricalAccuracy()])

    def plot_curves(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # summarize history for accuracy
        axs[0].plot(self.history.history['categorical_accuracy'])
        axs[0].plot(self.history.history['val_categorical_accuracy'])
        axs[0].set_title('model accuracy')
        axs[0].set_ylabel('accuracy')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'dev'], loc='upper left')
        axs[0].set_xticks(
            range(1, len(self.history.history['categorical_accuracy']) + 1, 4))

        # summarize history for loss
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('model loss')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'dev'], loc='upper right')
        axs[1].set_xticks(range(1, len(self.history.history['loss']) + 1, 4))

        # space between the plots
        plt.tight_layout()

        # show plot
        plt.show()
