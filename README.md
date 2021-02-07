# VPRNN
Official implementation and experimental results (excluding grid searches) for volume preserving recurrent neural networks (VPRNN) in keras with tensorflow 1.x backend.

## Dependencies
This repository depends on the `keras-vpnn` package. Install that with
```bash
pip install git+http://github.com/wtaylor17/keras-vpnn
```
The remaining dependencies (of the python package) are:
1. `wget`
2. `tensorflow` 1.x, tested with `1.15.2`
3. `keras` version `2.3.1`
4. `numpy`

Additionally for some scripts, you may need `matplotlib`, `seaborn`, `tqdm`, or some other utilities.
Some scripts import `keras_layer_normalization` (see [here](https://github.com/CyberZHG/keras-layer-normalization))
but the experiments reported don't actually use it.

## Installation
Run
```bash
pip install .
```
from the root directory of this repository, or alternatively
```bash
pip install git+http://github.com/wtaylor17/VPRNN
```
from anywhere with an internet connection.

## Experiments

1. Addition problem `T=500,1000,5000,10000`
2. Sequential MNIST classification
3. Permuted MNIST classification
4. IMDB movie review classification
5. HAR-2 human activity classification

For each experiment (other than the addition problem), 5 training runs with the best hyperparameters
were done after a grid search based on validation performance.
The logs and models for the first of each of these 5 runs is provided.

Single layer accuracies and parameters (IMDB excludes embeddings):

| Data Set | Test Accuracy (%) | Approx. Parameters |
|---|---|---|
| MNIST | 98.12 | 11k |
| pMNIST | 96.01 | 11k |
| IMDB | 87.74 | 56k |
| HAR-2 | 94.94 | 5k |