import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Input, Embedding,
                                     Bidirectional, GRU, TimeDistributed)
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import matplotlib.pyplot as plt

from evaluation_utils import (CustomPOSMetrics, EvaluationReport)
from constants import SEED

# Set tensorflow random seed for reproducibility
tf.random.set_seed(SEED)


class RNNPosTaggerTuner:
    def __init__(self, tuner_directory, project_name,
                 train_sentences, dev_sentences, data_processor,
                 embedding_matrix, n_stacked=3, train_size=0.5, seed=SEED):
        self.tuner_directory = tuner_directory
        self.project_name = project_name
        self.data_processor = data_processor
        self.embedding_matrix = embedding_matrix
        self.embedding_dimensions = embedding_matrix.shape[-1]
        self.n_stacked = n_stacked
        self.seed = seed

        self.num_classes = len(self.data_processor.idx2tag)

        train_X, train_y = data_processor.transform(train_sentences)
        sample_size = int(len(train_X) * train_size)

        self.train_X = train_X[0:sample_size+1]
        self.train_y = train_y[0:sample_size+1]

        dev_X, dev_y = data_processor.transform(dev_sentences)
        self.dev_X = dev_X
        self.dev_y = dev_y

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
                     validation_data=(self.dev_X, self.dev_y),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[early_stopping])

        return tuner

    def build_model(self, hp):
        model = Sequential()

        model.add(Input(shape=(self.data_processor.max_sequence_length,)))
        model.add(Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_dimensions,
                                 weights=[self.embedding_matrix],
                                 input_length=self.data_processor.max_sequence_length,
                                 mask_zero=True, trainable=False))

        model.add(Dropout(hp.Float(name='dropout_layer_first',
                  min_value=0.1, max_value=0.5, step=0.05)))

        num_layers = hp.Int('num_layers', min_value=1,
                            max_value=self.n_stacked, step=1)

        for i in range(0, num_layers):
            hp_gru_units = hp.Int('gru_units_'+str(i),
                                  min_value=64, max_value=256, step=32)
            hp_dropout = hp.Float(
                name='dropout_layer_'+str(i), min_value=0.1, max_value=0.5, step=0.05)

            model.add(Bidirectional(GRU(hp_gru_units, return_sequences=True)))
            model.add(Dropout(hp_dropout))

        dense_units = hp.Int(
            name='dense_units', min_value=64, max_value=256, step=32)
        model.add(Dense(units=dense_units, activation='relu'))
        model.add(Dropout(hp.Float(name='dropout_layer_last',
                  min_value=0.1, max_value=0.5, step=0.05)))

        model.add(TimeDistributed(
            Dense(self.num_classes, activation='softmax')))

        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=hp_learning_rate),
                           metrics=['categorical_accuracy'])

        return model


class RNNPosTagger:
    def __init__(self, models_dir, weights_name, data_processor, embedding_matrix):
        self.models_dir = models_dir
        self.weights_name = weights_name
        self.weights_path = os.path.join(self.models_dir, self.weights_name)
        self.data_processor = data_processor
        self.embedding_matrix = embedding_matrix
        self.embedding_dimensions = embedding_matrix.shape[-1]

        self.num_classes = len(self.data_processor.idx2tag)

        self.model = None
        self.history = None

    def fit(self, train_sentences, dev_sentences, hyperparams,
            batch_size=128, epochs=100, patience=10):
        train_X, train_y = self.data_processor.transform(train_sentences)
        dev_X, dev_y = self.data_processor.transform(dev_sentences)

        self.build_model(hyperparams)

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

        self.history = self.model.fit(train_X, train_y,
                                      validation_data=(dev_X, dev_y),
                                      batch_size=batch_size, epochs=epochs, shuffle=True,
                                      callbacks=[CustomPOSMetrics(valid_data=(dev_X, dev_y)),
                                                 checkpoint, early_stopping])

    def classification_report(self, dataset):
        dataset_X, dataset_y = self.data_processor.transform(dataset)
        predictions_proba = self.model.predict(dataset_X)
        predictions_2d = np.argmax(predictions_proba, axis=-1)

        y_true = []
        y_predicted = []
        y_probabilities = []

        # consolidate for each word
        for sequence_idx in range(predictions_proba.shape[0]):
            for word_idx in range(predictions_proba.shape[1]):
                # Check if the true label is not the padding token
                if np.argmax(dataset_y[sequence_idx][word_idx]) != 0:  
                    y_true.append(np.argmax(dataset_y[sequence_idx][word_idx]))
                    y_predicted.append(predictions_2d[sequence_idx][word_idx])
                    probabilities = predictions_proba[sequence_idx][word_idx]
                    y_probabilities.append(probabilities)

        # wrap results as numpy arrays
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        y_probabilities = np.array(y_probabilities)

        tag_ids = list(self.data_processor.idx2tag.keys())
        # reject the <PAD> (0) class
        class_ids = [tag_id for tag_id in tag_ids if tag_id > 0]
        class_labels = [self.data_processor.idx2tag[class_id] for class_id in class_ids]

        return EvaluationReport.classification_report(y_true, y_probabilities, y_predicted,
                                                      class_ids, class_labels)

    def plot_curves(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # summarize history for accuracy
        axs[0].plot(self.history.history['categorical_accuracy'])
        axs[0].plot(self.history.history['val_categorical_accuracy'])
        axs[0].set_title('Model Accuracy (Categorical Accuracy)')
        axs[0].set_ylabel('accuracy')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'dev'], loc='upper left')
        axs[0].set_xticks(
            range(1, len(self.history.history['categorical_accuracy'])+1, 4))

        # summarize history for loss
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('Model Loss (Categorical Cross-Entropy)')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'dev'], loc='upper right')
        axs[1].set_xticks(range(1, len(self.history.history['loss'])+1, 4))

        # # space between the plots
        plt.tight_layout()

        # show plot
        plt.show()

    def build_model(self, hyperparams):
        self.model = Sequential()

        self.model.add(Input(shape=(self.data_processor.max_sequence_length,)))
        self.model.add(Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_dimensions,
                                 weights=[self.embedding_matrix],
                                 input_length=self.data_processor.max_sequence_length,
                                 mask_zero=True, trainable=False))

        self.model.add(Dropout(hyperparams.get('dropout_layer_first')))

        n_stacked = hyperparams.get('num_layers')

        for i in range(0, n_stacked):
            gru_units = hyperparams.get("gru_units_{}".format(i))
            dropout = hyperparams.get("dropout_layer_{}".format(i))

            self.model.add(Bidirectional(
                GRU(gru_units, return_sequences=True)))
            self.model.add(Dropout(dropout))

        dense_units = hyperparams.get('dense_units')
        self.model.add(Dense(units=dense_units, activation='relu'))
        self.model.add(Dropout(hyperparams.get('dropout_layer_last')))

        self.model.add(TimeDistributed(
            Dense(self.num_classes, activation='softmax')))

        learning_rate = hyperparams.get('learning_rate')
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=learning_rate),
                           metrics=['categorical_accuracy'])

        return
