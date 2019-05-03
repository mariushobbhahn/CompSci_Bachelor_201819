#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${HOME}/Code/
python3 $1 --dataset="VOC" --normalize="False" --random="True" --sub_mean="False" 
