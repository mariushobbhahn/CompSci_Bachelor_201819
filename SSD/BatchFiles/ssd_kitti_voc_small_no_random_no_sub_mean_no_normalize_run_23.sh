#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${HOME}/Code/
python3 $1 --dataset="kitti_voc_small" --normalize="False" --random="False" --sub_mean="False" 