#!/bin/bash

#SBATCH --job-name=ssd_kitti_voc_no_random_sub_mean_normalize_

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1080ti:1

#SBATCH --partition=day
#SBATCH --time=23:59:59
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=ssd_kitti_voc_no_random_sub_mean_normalize_.err
#SBATCH --output=ssd_kitti_voc_no_random_sub_mean_normalize_.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=marius.hobbhahn@student.uni-tuebingen.de


srun -N1 -n1 -c $SLURM_CPUS_PER_TASK --gres=gpu:1080ti:1 singularity exec --nv /home/ s/Desktop/Comp.Sci_Bachelor_2018/SSD/BatchFiles/ssd_kitti_voc_no_random_sub_mean_normalize_run_12.sh s/Desktop/Comp.Sci_Bachelor_2018/SSD/train.py
echo DONE!
