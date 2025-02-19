"""A module for providing tuning and training capabilities for Multi-Layer Perceptron networks"""

import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import keras_tuner as kt
import matplotlib.pyplot as plt

from callback_metrics import Metrics
from evaluation_report import EvaluationMetrics

SEED = 2025

class MLPTuner:
    def __init__(self,
                 train_dataset, validation_dataset, tuner_directory, project_name,
                 train_size=0.5, seed=SEED, max_layers=3, use_dropout=True):
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.train_size = train_size
        self.seed = seed
        self.max_layers = max_layers
        self.use_dropout = use_dropout

        X_rem, X_train, y_rem, y_train = train_test_split(train_dataset[0].copy(), train_dataset[1].copy(),
                                                          stratify=train_dataset[1].copy(),
                                                          test_size=self.train_size,
                                                          random_state=self.seed)

        self.val_X, self.val_y = validation_dataset
        self.train_X = X_train
        self.train_y = y_train

    def tune(self, max_trials=10, epochs=30, batch_size=128, patience=5):
        tuner = kt.RandomSearch(self.build_model,
                                objective=kt.Objective("val_loss", direction="min"),
                                max_trials = max_trials,
                                seed=self.seed,
                                directory=self.tuner_directory,
                                project_name=self.project_name)

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        tuner.search(self.train_X, self.train_y,
                     validation_data=(self.val_X, self.val_y),
                     epochs=epochs, batch_size = batch_size,
                     callbacks=[early_stopping])

        return tuner

    def build_model(self, hp):
        model = Sequential()

        num_layers = hp.Int(name='num_layers', min_value=1, max_value=self.max_layers)

        for i in range(num_layers):
            units = hp.Int(name=f'hidden_units_{i}', min_value=64, max_value=256, step=64)
            activation = hp.Choice(name=f'activation_layer_{i}', values=['relu', 'tanh'])

            if self.use_dropout:
                dropout_rate = hp.Choice(name=f'dropout_layer_{i}', values=[0.1, 0.2, 0.3, 0.4, 0.5])

            if i == 0:
                model.add(Dense(units, activation=activation, input_dim=self.train_X.shape[1]))
            else:
                model.add(Dense(units, activation=activation))

            if self.use_dropout:
                model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(self.train_y.shape[1], activation='softmax'))

        # Optimizer tuning
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=hp_learning_rate),
                    metrics=[CategoricalAccuracy()])

        return model

class MLP:

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

        return EvaluationMetrics.classification_report(
            y_true, y_probabilities, y_predicted,
            self.class_ids, self.class_labels)

    def build_model(self, hyperparams, input_dim, output_dim):
        """Builds a configurable MLP model based on given hyperparameters."""

        self.model = Sequential()

        num_layers = hyperparams.get('num_layers', 1)  # Default to 1 layer if missing

        for i in range(num_layers):
            hidden_units = hyperparams.get(f'hidden_units_{i}', 64)  # Default to 64 neurons
            activation = hyperparams.get(f'activation_layer{i}', 'relu')  # Default activation: ReLU
            dropout_proba = hyperparams.get(f'dropout_layer_{i}', None)  # Default dropout: None

            # Add first layer with input_dim specified
            if i == 0:
                self.model.add(Dense(hidden_units, input_dim=input_dim, activation=activation,
                                     name=f'hidden_units_{i}'))
            else:
                self.model.add(Dense(hidden_units, activation=activation, name=f'hidden_units_{i}'))

            if dropout_proba is not None:
                # Add dropout layer
                self.model.add(Dropout(dropout_proba, name=f'dropout_layer_{i}'))

        # Output layer (Softmax for multi-class classification)
        self.model.add(Dense(output_dim, activation='softmax', name='output_layer'))

        # Compile model with Adam optimizer
        learning_rate = hyperparams.get('learning_rate', 0.001)  # Default learning rate: 0.001
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
        axs[0].set_xticks(range(1,len(self.history.history['categorical_accuracy']) + 1, 4))

        # summarize history for loss
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('model loss')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'dev'], loc='upper right')
        axs[1].set_xticks(range(1,len(self.history.history['loss']) + 1, 4))

        # space between the plots
        plt.tight_layout()

        # show plot
        plt.show()
