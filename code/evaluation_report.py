"""A module for evaluation"""

from sklearn.metrics import (f1_score, recall_score, precision_score,
                             auc, precision_recall_curve)
import pandas as pd
import numpy as np


class EvaluationMetrics:
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
        macro_average_df['Macro Average Precision Recall AUC'] = [np.mean(auc_scores)]

        return (classification_report_df, macro_average_df)
