#!/usr/bin/python
import os
import sys
from string import Template

BATCHFILE_TEMPLATE = "batchTemplate_eval.tmp"
RUNFILE_TEMPLATE = "runTemplate_eval_scattering_par.tmp"
FILEDIRECTORY = "BatchFiles/"
WORKPATH = os.path.dirname(os.path.realpath(sys.argv[0]))[11:]  # 11 to remove /mnt/beegfs
print("workpath: ", WORKPATH)
import argparse

# ADAPT:
# NETFILENAME = "train.py"


class Run_eval:
    """
    defines hyper parameters for one network
    """

    def __init__(self, weights_name, format_, random_aug, dataset, batch_norm, pretrained):
        self.weights_name = weights_name
        self.format = format_
        self.random_aug = random_aug
        self.dataset = str(dataset)
        self.batch_norm = batch_norm
        self.pretrained = pretrained

def apply_runs(runs):
    """
     starts a slurm job for each run to evaluate
    :param runs: list of tyoe :class:`Run`
    """
    if not os.path.exists(FILEDIRECTORY):
        os.makedirs(FILEDIRECTORY)

    for i in range(len(runs)):
        run = runs[i]
        with open(RUNFILE_TEMPLATE) as runb:
            file = runb.read()
        t = Template(file)
        runcontent = t.substitute(vars(run))
        runFilePath = FILEDIRECTORY + run.weights_name + "run_" + str(i) + ".sh"
        with open(runFilePath, "w+") as runfile:
            runfile.write(runcontent)
        os.system("chmod 775 " + runFilePath)
        with open(BATCHFILE_TEMPLATE) as batchTemp:
            file = batchTemp.read()
        t = Template(file)
        batchcontent = t.substitute(runfile=runFilePath, weights_name=run.weights_name, workpath=WORKPATH)
                                    #netfile=NETFILENAME)
        batch_file_path = FILEDIRECTORY + run.weights_name + str(i) + "_.sbatch"
        with open(batch_file_path, "w+") as batchfile:
            batchfile.write(batchcontent)
        os.system("chmod 775 " + batch_file_path)
        print("batch file path: ", batch_file_path)
        os.system("sbatch " + batch_file_path)


def build_run_parameter_grid(datasets, formats, random_augs, batch_norm, gen, pretrained):
    runs = []
    for d in datasets:
        if d == 'VOC':
            max_iter = 125000
        elif d == 'kitti_voc':
            max_iter = 200000
        elif d == 'kitti_voc_small':
            max_iter = 50000
	else:
            max_iter = 100000

        for r in random_augs:
            for f in formats:
                for b in batch_norm:
                    for p in pretrained:
                        weights_name = str('scattering_parallel_ssd_J2_' +
                            str(d) + '_' +
                            #'{}_'.format(f) +
                            '{}_'.format('random' if r else 'no_random') +
                            '{}_'.format('batch_norm' if b else 'no_batch_norm') +
                            '{}_'.format('pretrained' if p else 'no_pretrained') + 
                            '{}_'.format(str(gen)) +
                            '{}'.format(max_iter)
                        )
                        if f == '1000x300' and d in ['kitti_voc', 'kitti_voc_small'] or f == '300x300':
                            runs.append(Run_eval(weights_name=weights_name, dataset=d, format_=f, random_aug=r, batch_norm=b, pretrained=p))
    return runs




if __name__ == "__main__":
    """
    define your runs here
    """

    parser = argparse.ArgumentParser(
        description='eval grid for all ssds')

    parser.add_argument('--gen', default='13', type=str, help="generation of the currents test series")
    args = parser.parse_args()

    datasets = ['VOC', 'kitti_voc', 'toy_data', 'deformation_data', 'rotation_data', 'scale_data', 'translation_data']
    #datasets = ['kitti_voc']
    random_augs = [True]
    formats = ['300x300']
    batch_norms = [False]
    pretrained = [True]
    gen = '13.2'
    runs = build_run_parameter_grid(datasets=datasets, formats=formats, random_augs=random_augs, batch_norm=batch_norms, gen=gen, pretrained=pretrained)


    apply_runs(runs)

    print("started " + str(len(runs)) + " evaluation runs")

#5179 failed:x
