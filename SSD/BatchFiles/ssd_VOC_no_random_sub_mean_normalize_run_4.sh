#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${HOME}/Code/
python3 $1 --dataset="VOC" --normalize="True" --random="False" --sub_mean="True" 
