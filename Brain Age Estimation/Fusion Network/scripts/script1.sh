#!/bin/bash -l

# Request 16 CPUs
#$ -pe omp 16

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_type=A100

# Email when done
#$ -m ea

# Combine output and error files into a single file
#$ -j y

# 1 hour
#$ -l h_rt=48:00:00

python trainingreg_temp.py -d /projectnb/bucbi -s model_temp.pth