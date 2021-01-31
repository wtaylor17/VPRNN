import keras
import numpy as np
import wget
from zipfile import ZipFile

import os
import json


SCRIPT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, 'imdb-stash')
EMBEDDING_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, 'imdb_embeddings')


def stash_imdb(directory=DEFAULT_DIRECTORY, num_words=25000):
    os.makedirs(directory, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test)
    word_idx = keras.datasets.imdb.get_word_index()

    def joiner(s):
        return os.path.join(directory, s)
    np.save(joiner('x_train.npy'), x_train)
    np.save(joiner('y_train.npy'), y_train)
    np.save(joiner('x_test.npy'), x_test)
    np.save(joiner('y_test.npy'), y_test)
    with open(joiner('word_idx.json'), 'w') as fp:
        json.dump(word_idx, fp)


def load_imdb_stash(directory=DEFAULT_DIRECTORY):
    def joiner(s):
        return os.path.join(directory, s)
    return (np.load(joiner('x_train.npy')), np.load(joiner('y_train.npy'))),\
           (np.load(joiner('x_test.npy')), np.load(joiner('y_test.npy')))


def load_imdb_word_idx(directory=DEFAULT_DIRECTORY):
    def joiner(s):
        return os.path.join(directory, s)
    with open(joiner('word_idx.json'), 'r') as fp:
        return json.load(fp)


def stash_embeddings():
    os.makedirs(EMBEDDING_DIRECTORY, exist_ok=True)
    zip_path = os.path.join(EMBEDDING_DIRECTORY, 'glove.6B.zip')
    wget.download('http://nlp.stanford.edu/data/glove.6B.zip',
                  zip_path)
    with ZipFile(zip_path) as zf:
        zf.extractall(EMBEDDING_DIRECTORY)
    os.remove(zip_path)


def load_embeddings_stash(dim=100):
    assert dim in [50, 100, 200, 300]
    embeddings_path = os.path.join(EMBEDDING_DIRECTORY, f'glove.6B.{dim}d.txt')
    embeddings_index = {}
    with open(embeddings_path, errors='ignore') as fp:
        for line in fp.read().split('\n'):
            if not line:
                continue
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index


def create_embeddings_matrix(dim=100, num_words=25000):
    embeddings_index = load_embeddings_stash(dim=dim)
    word_idx = load_imdb_word_idx()
    embeddings_matrix = np.zeros((num_words, dim))
    hits, misses = 0, 0
    for word, i in word_idx.items():
        if word in embeddings_index and (num_words is None or i < num_words):
            embeddings_matrix[i] = embeddings_index[word]
            hits += 1
        elif i < num_words:
            misses += 1
    return embeddings_matrix, hits, misses
