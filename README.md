# VPRNN
Implementation and experimental results for volume preserving recurrent neural networks (VPRNN) in keras with tensorflow 1.x backend.

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