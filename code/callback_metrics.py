import tensorflow as tf
import numpy as np
from sklearn.metrics import (f1_score, recall_score, precision_score)

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
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        val_targ = tf.cast(val_targ,dtype=tf.float32)
        # If val_targ is 1-hot
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
          val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict,average="weighted")
        _val_recall = recall_score(val_targ, val_predict,average="weighted")
        _val_precision = precision_score(val_targ, val_predict,average="weighted")

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))

        return
