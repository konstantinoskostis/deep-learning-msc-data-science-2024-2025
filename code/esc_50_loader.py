import os
import pandas as pd

class ESC50Loader:
    """Loader of metadata dataframes"""

    def __init__(self):
        pass

    def load(self, path):
        """Load the training, validation and test datasets"""

        metadata_path = os.path.join(path, 'meta', 'esc50.csv')
        metadata_df = pd.read_csv(metadata_path, usecols=['filename', 'fold' , 'target', 'category'])

        train_df = metadata_df[metadata_df['fold'].isin([1, 2, 3])]
        val_df = metadata_df[metadata_df['fold'] == 4]
        test_df = metadata_df[metadata_df['fold'] == 5]

        train_df['audio_path'] = train_df['filename'].apply(lambda x: os.path.join(path, 'audio', x))
        val_df['audio_path'] = val_df['filename'].apply(lambda x: os.path.join(path, 'audio', x))
        test_df['audio_path'] = test_df['filename'].apply(lambda x: os.path.join(path, 'audio', x))

        return (train_df, val_df, test_df)
