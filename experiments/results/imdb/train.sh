#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=00-23:00
#SBATCH --output=%N-%j.out

module load cuda cudnn
source $HOME/scratch/tf1env/bin/activate

python $HOME/scratch/vpnn_repos/VPRNN/experiments/imdb/train.py --epochs 300 --model_name 1layer_imdb --hidden_dim 512\
     --lr 0.001 --n_layers 1 --cell vanilla --batch_size 128 --optimizer rmsprop --embedding_dropout 0.4 --patience 30 --test_mode
