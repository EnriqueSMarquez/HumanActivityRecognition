#!/bin/bash
cd /home/esm1g14/Documents/CNN_RelatedProjects/PhD_Experiments/SecondHalf/HumanActivityRecognition
module load cuda/
CUDA_VISIBLE_DEVICES=0 python main.py $SLURM_ARRAY_TASK_ID 0 &
CUDA_VISIBLE_DEVICES=1 python main.py $SLURM_ARRAY_TASK_ID 1 &
CUDA_VISIBLE_DEVICES=2 python main.py $SLURM_ARRAY_TASK_ID 2 &
CUDA_VISIBLE_DEVICES=3 python main.py $SLURM_ARRAY_TASK_ID 3 &
# CUDA_VISIBLE_DEVICES=2 python main.py $SLURM_ARRAY_TASK_ID 4 &
# CUDA_VISIBLE_DEVICES=2 python main.py $SLURM_ARRAY_TASK_ID 5 &
# CUDA_VISIBLE_DEVICES=3 python main.py $SLURM_ARRAY_TASK_ID 6 &
# CUDA_VISIBLE_DEVICES=3 python main.py $SLURM_ARRAY_TASK_ID 7
# CUDA_VISIBLE_DEVICES=3 python main.py $SLURM_ARRAY_TASK_ID 7
