#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${HOME}/Code/
python3 $1 --dataset="kitti_voc" --normalize="True" --random="True" --sub_mean="False" 
