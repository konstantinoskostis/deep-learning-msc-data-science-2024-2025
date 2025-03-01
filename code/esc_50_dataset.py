import numpy as np
from tensorflow.keras.utils import to_categorical


class ESC50Dataset:
    def __init__(self, df):
        self.df = df

        self.X_mel = []
        self.X_mfcc = []

        self.y = []

        self.class_ids = []
        self.class_labels = []

        self.compute_class_ids_and_labels()

    def compute_class_ids_and_labels(self):
        pairs = []

        for i, row in self.df.iterrows():
            self.y.append(row['target'])

            class_id = row['target']
            class_label = row['category']

            if (class_id, class_label) not in pairs:
                pairs.append((class_id, class_label))

        # sort by class_id
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        for pair in sorted_pairs:
            self.class_ids.append(pair[0])
            self.class_labels.append(pair[1])

        return

    def append_audio_features(self, x_mel, x_mfcc):
        self.X_mel.append(x_mel)
        self.X_mfcc.append(x_mfcc)

        return

    def to_numpy(self, one_hot_labels=True):
        # Add channel dimension for CN, to spectrograms
        self.X_mel_np = np.array(self.X_mel)[..., np.newaxis]

        # Shape (samples, time, features) for the MFCCs
        self.X_mfcc_np = np.array(self.X_mfcc).transpose(0, 2, 1)

        self.y_np = np.array(self.y)

        if one_hot_labels is True:
            self.y_np = to_categorical(
                self.y_np, num_classes=len(self.class_ids))

        return self
