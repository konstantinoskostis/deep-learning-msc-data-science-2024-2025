import os
import requests
from conllu import parse_incr
import gensim.downloader as api
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer


def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


class DatasetHandler:
    """ DatasetHandler

    Downloads and parses the universal-dependencies/en-gmu data files.
    """

    def __init__(self, data_directory, url, mode):
        self.url = url
        self.mode = mode

        self.sentences = []

        self.data_directory = data_directory
        self.data_file = os.path.join(
            self.data_directory, "{}.conllu".format(self.mode))

    def fetch(self):
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        if not os.path.exists(self.data_file):
            self.download()

        handle = open(self.data_file, "r", encoding="utf-8")
        for tokenlist in parse_incr(handle):
            sentence = []

            for token in tokenlist:
                if not token['form']:
                    continue
                sentence.append((token['form'].lower(), token['upos']))

            self.sentences.append(sentence)

    def basic_stats(self):
        n_sentences = len(self.sentences)

        words = []
        average_sentence_length = 0

        for sentence in self.sentences:
            average_sentence_length += len(sentence)

            for item in sentence:
                words.append(item[0])

        average_sentence_length /= n_sentences

        n_words = len(words)
        n_unique_words = len(set(words))

        stats = [[self.mode, n_sentences, round(
            average_sentence_length, 1), n_words, n_unique_words]]
        df = pd.DataFrame(stats, columns=[
                          'Dataset', 'Sentences', 'Average Sentence Length', 'Words', 'Unique Words'])

        return df

    def download(self):
        response = requests.get(self.url)
        with open(self.data_file, mode="wb") as file:
            file.write(response.content)


class SentenceUtils:
    """Sentence utility methods."""

    @staticmethod
    def tokens_of(tagged_sentence):
        return [token for (token, tag) in tagged_sentence]

    @staticmethod
    def pos_of(tagged_sentence):
        return [tag for (token, tag) in tagged_sentence]

    @staticmethod
    def texts(sentences):
        return [__class__.tokens_of(sentence) for sentence in sentences]

    @staticmethod
    def tags(sentences):
        return [__class__.pos_of(sentence) for sentence in sentences]

    @staticmethod
    def flatten_texts(sentences):
        return [token
                for tokenList in __class__.texts(sentences)
                for token in tokenList]

    @staticmethod
    def flatten_tags(sentences):
        return [tag
                for tagList in __class__.tags(sentences)
                for tag in tagList]


class DataProcessor:
    """ A data processor with a Scikit-Learn like interface.

    The class is responsible for processing a dataset, by using a Tokenizer
    on input texts, converting labels to ids (different Tokenizer) and
    padding sequences appropriately (and is used in the custom architecture)
    """

    def __init__(self, vocabulary_size, max_sequence_length):
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.word_tokenizer = Tokenizer(
            num_words=self.vocabulary_size, oov_token='__UNK__')
        self.tag_tokenizer = Tokenizer()

    def fit(self, dataset):
        X = SentenceUtils.texts(dataset)
        Y = SentenceUtils.tags(dataset)

        assert len(X) == len(Y)

        for idx in range(0, len(X)):
            assert len(X[idx]) == len(Y[idx])

        self.word_tokenizer.fit_on_texts(X)
        self.tag_tokenizer.fit_on_texts(Y)

        return self

    def transform(self, dataset):
        """ Transform the given dataset

        Encode and pad/truncate the input sequences (X)
        Encode the labels (id assignment)
        """

        X = SentenceUtils.texts(dataset)
        X_encoded = self.word_tokenizer.texts_to_sequences(X)

        Y = SentenceUtils.tags(dataset)
        Y_encoded = self.tag_tokenizer.texts_to_sequences(Y)

        X_padded = pad_sequences(
            X_encoded, maxlen=self.max_sequence_length, padding='post')
        Y_padded = pad_sequences(
            Y_encoded, maxlen=self.max_sequence_length, padding='post')

        return (X_padded, Y_padded)


class GensimEmbeddings:
    def __init__(self, data_processor, vocabulary_size, model_name='glove-wiki-gigaword-100'):
        self.data_processor = data_processor
        self.vocabulary_size = vocabulary_size
        self.embedding_model = api.load(model_name)
        self.embedding_dimensions = None
        self._embeddings_matrix = None

    def dimensions(self):
        if self.embedding_dimensions is None:
            self.embedding_dimensions = self.embedding_model.get_vector(
                'happy').shape[0]

        return self.embedding_dimensions

    def embeddings_matrix(self):
        if self._embeddings_matrix is None:
            embedding_matrix = np.zeros(
                shape=(self.vocabulary_size, self.dimensions()))

            for w2idx, _word in self.data_processor.word_tokenizer.index_word.items():
                # Skip PAD / UNK tokens
                if w2idx < 2:
                    continue
                try:
                    embedding_matrix[w2idx] = self.embedding_model[_word]
                except:
                    pass

            self._embeddings_matrix = embedding_matrix

        return self._embeddings_matrix


class PretrainedPosProcessor:
    def __init__(self, tokenizer, max_sequence_length=64):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def fit(self, dataset):
        """Fit label encoder on training dataset"""
        self.label_encoder = LabelEncoder()

        for tagged_sentence in dataset.sentences:
            pos_tags = SentenceUtils.pos_of(tagged_sentence)
            for pos_tag in pos_tags:
                self.label_encoder.fit(pos_tag)

    def transform(self, dataset):
        """Tokenize and pad the given dataset"""
        input_ids = []
        attention_masks = []
        label_ids = []

        for tagged_sentence in dataset.sentences:
            tokens = SentenceUtils.tokens_of(tagged_sentence)
            pos_tags = SentenceUtils.pos_of(tagged_sentence)

            encodings = self.tokenizer(tokens, truncation=True, padding='max_length',
                                       max_length=self.max_length, is_split_into_words=True)
            input_ids.append(encodings['input_ids'])
            attention_masks.append(encodings['attention_mask'])
            label_ids.append(self.label_encoder.transform(pos_tags))
