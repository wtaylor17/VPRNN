#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=00-52:00
#SBATCH --output=%N-%j.out

module load cuda cudnn
source $HOME/scratch/tf1env/bin/activate

python $HOME/scratch/vpnn_repos/VPRNN/experiments/sequentialmnist/train_sequential_mnist.py --epochs 300 --stamp 1layer_smnist --dim 512\
     --activation relu --learning_rate 0.001 --tensorboard --n_layers 1 --rnn_type vanilla --batch_size 256 --optimizer rmsprop --nodiag --noclip\
     --input_dropout 0.1 --verbose 2 --test_mode
