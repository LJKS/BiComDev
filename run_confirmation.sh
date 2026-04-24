#!/bin/bash
#SBATCH --time=2-00:00
#SBATCH --mem=200gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
#SBATCH --mail-user=lschmid@uos.de
#SBATCH --mail-type=ALL
source $HOME/.bashrc
spack load miniconda3@4.10.3
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/cuda-11.8.0-x32erfzo6xl2qgbp5enezl53wwiingmt
export CUDA_DIR=/appl/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/cuda-11.8.0-x32erfzo6xl2qgbp5enezl53wwiingmt
source avtivate Bidirectional_CommGame
python ppo.py --num_same 2 --num_diff1 2 --num_diff2 2 --msg_len 1 --vocab_size 20  --num_steps 20 --num_epochs 8000 --embed_dim 48 --batch_size 128 --lstm_units 96
python ppo.py --num_same 2 --num_diff1 2 --num_diff2 2 --msg_len 1 --vocab_size 20  --num_steps 20 --num_epochs 8000 --embed_dim 48 --batch_size 128 --lstm_units 96
python ppo.py --num_same 2 --num_diff1 2 --num_diff2 2 --msg_len 1 --vocab_size 20  --num_steps 20 --num_epochs 8000 --embed_dim 48 --batch_size 128 --lstm_units 96
python ppo.py --num_same 2 --num_diff1 2 --num_diff2 2 --msg_len 1 --vocab_size 20  --num_steps 20 --num_epochs 8000 --embed_dim 48 --batch_size 128 --lstm_units 192 --which_agent separated
python ppo.py --num_same 2 --num_diff1 2 --num_diff2 2 --msg_len 1 --vocab_size 20  --num_steps 20 --num_epochs 8000 --embed_dim 48 --batch_size 128 --lstm_units 192 --which_agent separated
python ppo.py --num_same 2 --num_diff1 2 --num_diff2 2 --msg_len 1 --vocab_size 20  --num_steps 20 --num_epochs 8000 --embed_dim 48 --batch_size 128 --lstm_units 192 --which_agent separated