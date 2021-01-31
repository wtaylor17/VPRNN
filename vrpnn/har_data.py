import numpy as np
import wget
import shutil
from zipfile import ZipFile

import os


SCRIPT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, 'har-stash')


def stash_har(directory=DEFAULT_DIRECTORY):
    os.makedirs(directory, exist_ok=True)
    zip_path = os.path.join(directory, 'har.zip')
    unzipped_path = os.path.join(directory, 'har_raw')
    wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip',
                  zip_path)

    with ZipFile(zip_path) as zf:
        zf.extractall(unzipped_path)
    os.remove(zip_path)

    test_path = os.path.join(unzipped_path, "UCI HAR Dataset", "test")
    test_feature_path = os.path.join(test_path, "Inertial Signals")
    train_path = os.path.join(unzipped_path, "UCI HAR Dataset", "train")
    train_feature_path = os.path.join(train_path, "Inertial Signals")
    x_train_features, y_train = [], []
    x_test_features, y_test = [], []

    for feature_path in os.listdir(train_feature_path):
        with open(os.path.join(train_feature_path, feature_path), 'r') as fp:
            arr = []
            for line in fp:
                arr.append(list(map(float, [t for t in line.split() if t])))
            x_train_features.append(np.array(arr).reshape((-1, 128, 1)))
    x_train = np.concatenate(x_train_features, axis=-1)

    for feature_path in os.listdir(test_feature_path):
        with open(os.path.join(test_feature_path, feature_path), 'r') as fp:
            arr = []
            for line in fp:
                arr.append(list(map(float, [t for t in line.split() if t])))
            x_test_features.append(np.array(arr).reshape((-1, 128, 1)))
    x_test = np.concatenate(x_test_features, axis=-1)

    y_train = []
    with open(os.path.join(train_path, 'y_train.txt'), 'r') as fp:
        for line in fp:
            label = int(line.strip())
            if label in [2, 4, 6]:
                y_train.append(0)
            else:
                y_train.append(1)
    y_train = np.array(y_train).reshape((-1, 1))

    y_test = []
    with open(os.path.join(test_path, 'y_test.txt'), 'r') as fp:
        for line in fp:
            label = int(line.strip())
            if label in [2, 4, 6]:
                y_test.append(0)
            else:
                y_test.append(1)
    y_test = np.array(y_test).reshape((-1, 1))

    shutil.rmtree(unzipped_path)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    np.save(os.path.join(directory, 'x_train.npy'), x_train)
    np.save(os.path.join(directory, 'x_test.npy'), x_test)
    np.save(os.path.join(directory, 'y_train.npy'), y_train)
    np.save(os.path.join(directory, 'y_test.npy'), y_test)


def load_har_stash(directory=DEFAULT_DIRECTORY):
    return ((np.load(os.path.join(directory, 'x_train.npy')),
             np.load(os.path.join(directory, 'y_train.npy'))),
            (np.load(os.path.join(directory, 'x_test.npy')),
             np.load(os.path.join(directory, 'y_test.npy'))))
