"""A module containing evaluation utilities"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             auc, precision_recall_curve)


class Metrics(tf.keras.callbacks.Callback):
    """A sublcass if keras callback class for logging extra metrics.

    Per epoch the class records:
        - val_f1
        - val_recall
        - val_precision
    """

    def __init__(self, valid_data):
        """Initializes the Metrics class

        Args:
            valid_data (tuple): The validation dataset given as a tuple (val_X, val_y)
        """
        super(Metrics, self).__init__()

        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        """Recording of metrics on epoch end.

        Args:
            epoch (int): The number of the current epoch
            logs (dict, optional): The dictionary that stores the metrics per epoch
        """
        logs = logs or {}
        val_predict = np.argmax(
            self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        val_targ = tf.cast(val_targ, dtype=tf.float32)

        # If val_targ is 1-hot
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average="weighted")
        _val_recall = recall_score(val_targ, val_predict, average="weighted")
        _val_precision = precision_score(
            val_targ, val_predict, average="weighted")

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" %
              (_val_f1, _val_precision, _val_recall))

        return

class CustomPOSMetrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        self.X_val = valid_data[0]
        self.y_val = valid_data[1]
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_pred = self.model.predict(self.X_val)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(self.y_val, axis=-1)
        
        y_pred_flat, y_true_flat = [], []
        for pred_seq, true_seq in zip(y_pred, y_true):
            for pred, true in zip(pred_seq, true_seq):
                if true != 0:  # Exclude padding
                    y_pred_flat.append(pred)
                    y_true_flat.append(true)

        _val_f1 = f1_score(y_true_flat, y_pred_flat, average="weighted")
        _val_recall = recall_score(y_true_flat, y_pred_flat, average="weighted")
        _val_precision = precision_score(
            y_true_flat, y_pred_flat, average="weighted")

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision

        print(" — val_f1: %f — val_precision: %f — val_recall: %f" %
              (_val_f1, _val_precision, _val_recall))


class EvaluationReport:
    """A class for evaluating a model's responses"""

    @staticmethod
    def classification_report(y_true, y_probabilities, y_predicted, class_ids, class_labels):
        """ Creates a classification report for a multi-class classsification problem.

        Note: Scikit-Learn's precision_recall_curve has its output reversed. See:
        https://github.com/scikit-learn/scikit-learn/issues/2097

        Args:
            y_true: A one dimensional array(n_samples), containing the actual class id per sample.
            y_probabilities: A 2D array (n_samples, n_classes) containing the predicted probabilities
                per sample.
            y_predicted: A one dimensional array(n_samples) containing the predicted class id per sample.
            class_ids: A one dimensional array (n_classes) containing the ids of classes.
            class_labels: A one dimensional array (n_classes) containing the names of the classes.

        Returns:
            Tuple: Containing 2 pandas.DataFrame objects
        """

        # Compute precision (per class)
        precision = precision_score(y_true, y_predicted, average=None)

        # Compute recall (per class)
        recall = recall_score(y_true, y_predicted, average=None)

        # Compute F1 (per class)
        f1 = f1_score(y_true, y_predicted, average=None)

        # Compute Precision-Recall AUC score (per class)
        auc_scores = []
        for class_id in class_ids:
            class_indices = (y_true == class_id)
            if any(class_indices):
                class_precision, class_recall, thresholds = precision_recall_curve(
                    class_indices.astype(int), y_probabilities[:, class_id])
                class_precision_recall_auc = auc(class_recall, class_precision)
                auc_scores.append(class_precision_recall_auc)

        classification_report_df = pd.DataFrame()
        classification_report_df['Class Id'] = class_ids
        classification_report_df['Class Name'] = class_labels
        classification_report_df['Precision'] = precision
        classification_report_df['Recall'] = recall
        classification_report_df['F1'] = f1
        classification_report_df['Precision-Recall AUC'] = auc_scores

        macro_average_df = pd.DataFrame()
        macro_average_df['Macro Average Precision'] = [np.mean(precision)]
        macro_average_df['Macro Average Recall'] = [np.mean(recall)]
        macro_average_df['Macro Average F1'] = [np.mean(f1)]
        macro_average_df['Macro Average Precision Recall AUC'] = [
            np.mean(auc_scores)]

        return (classification_report_df, macro_average_df)
