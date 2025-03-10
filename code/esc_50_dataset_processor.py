import librosa
import numpy as np
from tqdm.auto import tqdm

from constants import SAMPLE_RATE, DURATION, N_MELS, N_MFCC, HOP_LENGTH


class ESC50DatasetProcessor:
    """Performs feature extraction on a given ESC-50 data slice"""

    def __init__(self,
                 sample_rate=SAMPLE_RATE, duration=DURATION,
                 n_mels=N_MELS, n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    def process(self, dataset,
                normalize_spectrograms=True, normalize_mfcc=True):
        """Process a ESC-50 dataset slice by extracting audio features"""

        for i, row in tqdm(dataset.df.iterrows(), total=len(dataset.df)):
            audio_features = self.extract_features(
                row['audio_path'],
                normalize_spectrograms=normalize_spectrograms,
                normalize_mfcc=normalize_mfcc
            )
            dataset.append_audio_features(*audio_features)

        return dataset

    def extract_features(self, audio_file_path,
                         normalize_spectrograms=True, normalize_mfcc=True):
        """Extract audio features (Mel's spectogram and MFCCs)"""

        # Load audio
        audio, sr = librosa.load(audio_file_path, sr=self.sample_rate,
                                 duration=self.duration, mono=True)

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB

        if normalize_spectrograms is True:
            mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr,
                                    n_mfcc=self.n_mfcc,
                                    hop_length=self.hop_length)

        if normalize_mfcc is True:
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        return mel_spec, mfcc
