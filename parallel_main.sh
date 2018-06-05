#!/bin/bash
# cd /home/esm1g14/Documents/CNN_RelatedProjects/PhD_Experiments/SecondHalf/HumanActivityRecognition
# module load cuda/
for ((i=0;i<=20;i++));
do
    echo "STARTING PARALLEL RUN $i";
    CUDA_VISIBLE_DEVICES=0 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 0 &
    CUDA_VISIBLE_DEVICES=0 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 1 &
    CUDA_VISIBLE_DEVICES=1 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 2 &
    CUDA_VISIBLE_DEVICES=1 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 3 &
    CUDA_VISIBLE_DEVICES=2 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 4 &
    CUDA_VISIBLE_DEVICES=2 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 5 &
    CUDA_VISIBLE_DEVICES=3 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 6 &
    CUDA_VISIBLE_DEVICES=3 python main.py -d pamap2 --max_scale 1 --parallel_index1 $i --parallel_index2 7
done
# SLURM_ARRAY_TASK_ID
