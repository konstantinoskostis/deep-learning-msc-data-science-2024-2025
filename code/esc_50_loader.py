import os
import pandas as pd

from constants import ESC_50_AUDIO_PATH, ESC_50_META_CSV_PATH


class ESC50Loader:
    """Loader of metadata dataframes"""

    def __init__(self, path):
        self.path = path

    def load(self):
        """Load the training, validation and test datasets"""

        metadata_path = os.path.join(self.path, ESC_50_META_CSV_PATH)
        metadata_df = pd.read_csv(metadata_path, usecols=[
                                  'filename', 'fold', 'target', 'category'])

        # subtract 1 from target so the class ids start from 0
        metadata_df['target'] = metadata_df['target'] - 1

        train_df = metadata_df[metadata_df['fold'].isin([1, 2, 3])]
        val_df = metadata_df[metadata_df['fold'] == 4]
        test_df = metadata_df[metadata_df['fold'] == 5]

        train_df['audio_path'] = train_df['filename'].apply(
            lambda x: os.path.join(self.path, ESC_50_AUDIO_PATH, x))
        val_df['audio_path'] = val_df['filename'].apply(
            lambda x: os.path.join(self.path, ESC_50_AUDIO_PATH, x))
        test_df['audio_path'] = test_df['filename'].apply(
            lambda x: os.path.join(self.path, ESC_50_AUDIO_PATH, x))

        return (train_df, val_df, test_df)
