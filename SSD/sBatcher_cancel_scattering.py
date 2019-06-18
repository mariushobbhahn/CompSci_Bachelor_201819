#!/usr/bin/python
import os
import sys
from string import Template
from sBatcher_train_scattering import build_run_parameter_grid, Run

BATCHFILE_TEMPLATE = "batchTemplate.tmp"
RUNFILE_TEMPLATE = "runTemplate_scattering.tmp"
FILEDIRECTORY = "BatchFiles/"
WORKPATH = os.path.dirname(os.path.realpath(sys.argv[0]))[11:]  # 11 to remove /mnt/beegfs
print("workpath: ", WORKPATH)


def apply_runs(runs):
    """
     starts a slurm job for each run
    :param runs: list of tyoe :class:`Run`
    """
    if not os.path.exists(FILEDIRECTORY):
        os.makedirs(FILEDIRECTORY)

    for i in range(len(runs)):
        run = runs[i]
        batch_file_path = run.experiment_name
        print("job to be cancelled: ", batch_file_path)
        os.system("scancel -n " + batch_file_path)



if __name__ == "__main__":
    """
    define your runs here
    """

    datasets = ['VOC', 'kitti_voc', 'toy_data', 'deformation_data', 'rotation_data', 'scale_data', 'translation_data']
    formats = ['300x300']
    random_augs = [True]
    batch_norms = [True]
    pretrained = [True, False]
    runs = build_run_parameter_grid(datasets=datasets, formats=formats, random_augs=random_augs, batch_norm=batch_norms, pretrained=pretrained)

    apply_runs(runs)

    print("cancelled " + str(len(runs)) + " runs")

#5179 failed
