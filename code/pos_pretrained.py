import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from transformers import  BertTokenizerFast, TFBertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

from evaluation_utils import EvaluationReport

from constants import SEED

# Set tensorflow random seed for reproducibility
tf.random.set_seed(SEED)

def fetch_tokenizer(bert_model_name='bert-base-uncased'):
    return BertTokenizerFast.from_pretrained(bert_model_name)

class LabelExtractor(LabelEncoder):
    def __init__(self):
        self.class_ids = []
        self.class_labels = []
        self.num_labels = 0

    def extract(self, dataset):
        if self.class_ids:
            return self

        unique_pos_tags = set()
        for sentence in dataset:
            for _, pos in sentence:
                unique_pos_tags.add(pos)

        self.fit(list(unique_pos_tags))

        self.class_labels = self.classes_
        self.class_ids = self.transform(self.class_labels)
        self.num_labels = len(self.class_labels)

        return self

# Processor class to handle tokenization, padding, and label alignment
class Processor:
    def __init__(self, tokenizer, label_encoder, max_length=128):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def preprocess_data(self, data):
        input_ids = []
        attention_masks = []
        labels = []

        for sentence in data:
            words = [word for word, _ in sentence]
            pos_tags = [self.label_encoder.transform([pos])[0] for _, pos in sentence]  # Encode POS tags

            encoding = self.tokenizer(words, is_split_into_words=True, padding='max_length',
                truncation=True, max_length=self.max_length, return_tensors='np')

            # Align labels with tokens (handling subword tokens and padding)
            aligned_labels = self.align_labels_with_tokens(encoding, pos_tags)

            input_ids.append(encoding['input_ids'][0])
            attention_masks.append(encoding['attention_mask'][0])
            labels.append(aligned_labels)

        return np.array(input_ids), np.array(attention_masks), np.array(labels)

    def align_labels_with_tokens(self, encoding, pos_tags):
        """
        Align labels with subword tokens by adding -100 for subword tokens
        to ignore them in the loss computation.
        """
        word_ids = encoding.word_ids()  # Get word_ids for alignment
        aligned_labels = []

        for i in range(len(word_ids)):
            if word_ids[i] is None:  # Special tokens (e.g., [CLS], [SEP])
                aligned_labels.append(-100)
            else:
                # If this is the first subword token, keep the label of the original word
                if word_ids[i] < len(pos_tags):
                    aligned_labels.append(pos_tags[word_ids[i]])  # Copy the label from the original word
                else:
                    aligned_labels.append(-100)  # For any additional tokens, we set -100 to ignore them

        # Pad or truncate the label sequence to match the model's input length
        aligned_labels = aligned_labels[:self.max_length]  # truncate if necessary
        aligned_labels += [-100] * (self.max_length - len(aligned_labels))  # pad if necessary

        return aligned_labels


# BertWithMLPHead class to define the BERT model with an MLP head
class BertWithMLPHead(tf.keras.Model):
    def __init__(self, num_labels, bert_model_name='bert-base-uncased',
                unfreeze_layers=3, mlp_units=[512, 256, 128], dropout_rate=0.3):
        super(BertWithMLPHead, self).__init__()

        self.unfreeze_layers = unfreeze_layers
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate

        # Load BERT model
        self.bert = TFBertModel.from_pretrained(bert_model_name)

        # Add an MLP head (fully connected layers with dropout)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.mlp_layers = []

        for units in self.mlp_units:
            self.mlp_layers.append(tf.keras.layers.Dense(units, activation='relu'))
            self.mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

         # No softmax here, just logits :-)
        self.dense_out = tf.keras.layers.Dense(num_labels, activation=None)

        # Unfreeze the last unfreeze_layers of BERT
        self.unfreeze_last_bert_layers()

    def call(self, inputs):
        # Extract the hidden states from BERT
        bert_output = self.bert(inputs)[0]  # (batch_size, seq_length, hidden_size)

        # Pass the output through the MLP head with dropout
        for i, layer in enumerate(self.mlp_layers):
            if i == 0:
                x = layer(bert_output)
            else:
                x = layer(x)

        # Apply the final dense layer to get the logits
        logits = self.dense_out(x)

        # Ensure that the logits have the shape [batch_size, sequence_length, num_labels]
        return logits

    def unfreeze_last_bert_layers(self):
        # Freeze all layers of the BERT model first
        for layer in self.bert.layers:
            layer.trainable = False

        # Unfreeze the last unfreeze_layers of BERT
        for layer in self.bert.layers[-self.unfreeze_layers:]:
            layer.trainable = True

# Custom Callback to monitor Precision, Recall, and F1-score
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(MetricsCallback, self).__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        # Predictions and ground truth for the entire validation set
        y_true = []
        y_pred = []

        for batch_inputs, batch_labels in self.val_data:
            # Get model predictions (logits)
            logits = self.model.predict(batch_inputs)

            # Convert to probabilities (using softmax)
            probabilities = tf.nn.softmax(logits).numpy()

            # Convert probabilities to predicted classes (indices of max logits)
            predicted_classes = np.argmax(probabilities, axis=-1)

            # Mask out the -100 labels (padding tokens)
            mask = batch_labels != -100

            # Use tf.reshape (because of EagerTensor)
            y_true.append(tf.reshape(batch_labels[mask], [-1]))  # Flatten using tf.reshape
            y_pred.append(tf.reshape(predicted_classes[mask], [-1])) # Flatten using tf.reshape

        # Concatenate all elements of y_true and y_pred into 1D arrays
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate metrics only for non -100 labels
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Log the metrics in the training logs
        logs['val_precision'] = precision
        logs['val_recall'] = recall
        logs['val_f1'] = f1

        print(f"\nEpoch {epoch + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Trainer class to handle training, evaluation, and reporting
class Trainer:
    def __init__(self, models_dir, weights_name, model, label_encoder,
                 learning_rate=5e-5, epochs=5):
        self.models_dir = models_dir
        self.weights_name = weights_name
        self.weights_path = os.path.join(self.models_dir, self.weights_name)

        self.model = model
        self.label_encoder = label_encoder

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.history = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define the loss
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compile_model(self):
        """Comile the model

        Use custom loss and accuracy functions to account fot -100 tokens.
        Use tf functions here because we deal with EagerTensor objects.
        """
        # Custom loss to handle -100 tokens
        def masked_loss(y_true, y_pred):
            mask = tf.math.logical_not(tf.math.equal(y_true, -100))  # Create a mask for padded tokens

            # Apply the mask to y_pred and y_true
            masked_y_pred = tf.boolean_mask(y_pred, mask)
            masked_y_true = tf.boolean_mask(y_true, mask)

            loss_ = self.loss(masked_y_true, masked_y_pred)  # Calculate the loss for non-padded tokens

            # Calculate the total loss for non-padded tokens and divide by the number of non-padded tokens
            reduced_masked_loss = tf.reduce_sum(loss_) / tf.cast(tf.shape(masked_y_true)[0], dtype=loss_.dtype)

            return reduced_masked_loss

        # Custom accuracy to handle -100 tokens
        def masked_accuracy(y_true, y_pred):
            # Get the predicted labels from logits
            predicted_labels = tf.argmax(y_pred, axis=-1)

            # Create a mask for valid tokens (those that are not padding)
            mask = tf.not_equal(y_true, -100)

            # Cast predicted_labels to the same type as y_true (float32)
            predicted_labels = tf.cast(predicted_labels, dtype=y_true.dtype)  # Cast to float32

            # Compare the true labels and predicted labels, applying the mask to ignore padding tokens
            correct_predictions = tf.equal(predicted_labels, y_true)

            # Apply the mask to only count valid tokens
            correct_predictions = tf.cast(correct_predictions, dtype=tf.float32) * tf.cast(mask, dtype=tf.float32)

            # Calculate accuracy: sum of correct predictions divided by the number of valid tokens
            accuracy = tf.reduce_sum(correct_predictions) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

            return accuracy

        # Use the masked loss function and masked accuracy during compilation
        self.model.compile(optimizer=self.optimizer, loss=masked_loss,
                           metrics=[masked_accuracy])

    def fit_model(self, train_data, val_data, patience=2):
        self.train_data = train_data
        self.val_data = val_data

        callbacks = []

        metrics_callback = MetricsCallback(val_data=self.val_data)

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

        callbacks.append(metrics_callback)
        callbacks.append(checkpoint)
        callbacks.append(early_stopping)

        self.history = self.model.fit(self.train_data, validation_data=self.val_data,
                                      shuffle=True, epochs=self.epochs, callbacks=callbacks)

    def classification_report(self, dataset):
        y_true = []
        y_predicted = []
        y_probabilities = []

        for (batch_inputs, batch_labels) in dataset:
            # Get predicted logits using predict
            logits = self.model.predict(batch_inputs)

            # Convert logits to probabilities using softmax (3D matrix)
            probabilities = tf.nn.softmax(logits).numpy()

            # Convert probabilities to class labels (0, 1, 2...)
            # 2D matrix
            predictions = np.argmax(probabilities, axis=-1)

            # consolidate for each word
            for sequence_idx in range(probabilities.shape[0]):
                for word_idx in range(probabilities.shape[1]):
                    if batch_labels[sequence_idx][word_idx] != -100:
                        y_true.append(batch_labels[sequence_idx][word_idx])
                        y_predicted.append(predictions[sequence_idx][word_idx])
                        y_probabilities.append(probabilities[sequence_idx][word_idx])

        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        y_probabilities = np.array(y_probabilities)

        return EvaluationReport.classification_report(
            y_true, y_probabilities, y_predicted,
            self.label_encoder.class_ids,
            self.label_encoder.class_labels)

    def plot_curves(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # summarize history for accuracy
        axs[0].plot(self.history.history['masked_accuracy'])
        axs[0].plot(self.history.history['val_masked_accuracy'])
        axs[0].set_title('Model Accuracy (Masked Accuracy)')
        axs[0].set_ylabel('accuracy')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'dev'], loc='upper left')
        axs[0].set_xticks(
            range(1, len(self.history.history['masked_accuracy'])+1, 4))

        # summarize history for loss
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('Model Loss (Sparse Categorical Cross-Entropy [Masked])')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'dev'], loc='upper right')
        axs[1].set_xticks(range(1, len(self.history.history['loss'])+1, 4))

        # # space between the plots
        plt.tight_layout()

        # show plot
        plt.show()
