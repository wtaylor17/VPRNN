#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=07-00:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
source $HOME/scratch/tf1env/bin/activate

# T = 500
python adding_problem_train.py --steps 500 --stamp vprelu500 --lr 0.002 --model vanilla --rotations 7 --epochs 200 --tensorboard --nodiag --noclip

# T = 1000
python adding_problem_train.py --steps 1000 --stamp vprelu1000 --lr 0.002 --model vanilla --rotations 7 --epochs 200 --tensorboard --nodiag --noclip

# T = 5000
python adding_problem_train.py --steps 5000 --stamp vprelu5000 --lr 0.002 --model vanilla --rotations 7 --epochs 200 --tensorboard --nodiag --noclip

# T = 10000
python adding_problem_train.py --steps 10000 --stamp vprelu10000 --lr 0.002 --model vanilla --rotations 7 --epochs 200 --tensorboard --nodiag --noclip
