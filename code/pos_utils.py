import os
import requests
from conllu import parse_incr
import gensim.downloader as api
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
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


class DatasetProcessor:
    """A data processor with a Scikit-Learn like interface.

    The class is responsible for processing a dataset. It persists a vocabulary
    from the training set and accounts for PAD and UKN tokens along with a mapping
    for tags/classes. Transformation includes encoding the input sequences and
    padding them. The classes are being one-hot encoded.
    (The class is used in the custom architecture)
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.max_sequence_length = None

    def fit(self, dataset):
        sentences = SentenceUtils.texts(dataset)
        tags = SentenceUtils.tags(dataset)
  
        self.word2idx = {word: idx + 2 for idx, word in enumerate(set(word for sentence in sentences for word in sentence))}
        self.word2idx["<PAD>"] = 0
        self.word2idx["<UNK>"] = 1
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.tag2idx = {tag: idx + 1 for idx, tag in enumerate(set(tag for sentence_tags in tags for tag in sentence_tags))}
        self.tag2idx["<PAD>"] = 0
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
    
        self.max_sequence_length = max(len(sentence) for sentence in sentences)

    def transform(self, dataset):
        sentences = SentenceUtils.texts(dataset)
        tags = SentenceUtils.tags(dataset)
  
        X = [[self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence] for sentence in sentences]
        X_padded = pad_sequences(X, maxlen=self.max_sequence_length, padding='post', value=self.word2idx["<PAD>"])

        y = [[self.tag2idx[tag] for tag in sentence_tags] for sentence_tags in tags]
        y_padded = pad_sequences(y, maxlen=self.max_sequence_length, padding='post', value=self.tag2idx["<PAD>"])
        y_categorized = np.array([to_categorical(tag_seq, num_classes=len(self.tag2idx)) for tag_seq in y_padded])
    
        return (X_padded, y_categorized)


class GloveEmbeddings:
    """A class for loading and managing GloVe embeddings."""

    def __init__(self, word_index, model_name="glove-wiki-gigaword-100"):
        self.glove_embeddings = api.load(model_name)
        self.word_index = word_index
        self.embedding_dim = self.glove_embeddings.vector_size
        self.embedding_matrix = self.load_embeddings()

    def load_embeddings(self):        
        embedding_matrix = np.zeros((len(self.word_index), self.embedding_dim))
        all_embeddings = []

        for word, i in self.word_index.items():
            if word in self.glove_embeddings:
                embedding = self.glove_embeddings[word]
                embedding_matrix[i] = embedding
                all_embeddings.append(embedding)
            elif word == "<UNK>":
                continue
        
        if all_embeddings:
            embedding_matrix[self.word_index["<UNK>"]] = np.mean(all_embeddings, axis=0)
        
        return embedding_matrix


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
