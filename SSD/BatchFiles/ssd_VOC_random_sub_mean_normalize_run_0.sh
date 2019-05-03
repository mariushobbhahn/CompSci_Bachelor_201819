#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${HOME}/Code/
python3 $1 --dataset="VOC" --normalize="True" --random="True" --sub_mean="True" 
