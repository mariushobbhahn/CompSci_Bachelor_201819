#!/usr/bin/python
import os
import sys
from string import Template

BATCHFILE_TEMPLATE = "batchTemplate_week.tmp"
RUNFILE_TEMPLATE = "runTemplate_scattering.tmp"
FILEDIRECTORY = "BatchFiles/"
WORKPATH = os.path.dirname(os.path.realpath(sys.argv[0]))[11:]  # 11 to remove /mnt/beegfs
print("workpath: ", WORKPATH)

# ADAPT:
# NETFILENAME = "train.py"


class Run:
    """
    defines hyper parameters for one network
    """

    def __init__(self, name, dataset, format, random_aug, batch_norm, pretrained):
        self.experiment_name = name
        self.dataset = str(dataset)
        self.format = str(format)
        self.random_aug = random_aug
        self.batch_norm = batch_norm
        self.pretrained = pretrained
        

def apply_runs(runs):
    """
     starts a slurm job for each run
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
        runFilePath = FILEDIRECTORY + run.experiment_name + "run_" + str(i) + ".sh"
        with open(runFilePath, "w+") as runfile:
            runfile.write(runcontent)
        os.system("chmod 775 " + runFilePath)
        with open(BATCHFILE_TEMPLATE) as batchTemp:
            file = batchTemp.read()
        t = Template(file)
        batchcontent = t.substitute(runfile=runFilePath, experiment_name=run.experiment_name, workpath=WORKPATH)
                                    #netfile=NETFILENAME)
        batch_file_path = FILEDIRECTORY + run.experiment_name + str(i) + ".sbatch"
        with open(batch_file_path, "w+") as batchfile:
            batchfile.write(batchcontent)
        os.system("chmod 775 " + batch_file_path)
        print("batch file path: ", batch_file_path)
        os.system("sbatch " + batch_file_path)


def build_run_parameter_grid(datasets, formats, random_augs, batch_norm, pretrained):
    runs = []
    for d in datasets:
        for f in formats:
            for r in random_augs:
                for b in batch_norm:
                    for p in pretrained:
                        experiment_name = str('scattering_ssd_J2_' +
                            str(d) + '_' +
                            str(f) + '_' +
                            '{}'.format('random_' if r else 'no_random_') +
                            '{}'.format('batch_norm_' if b else 'no_batch_norm_') +  
                            '{}'.format('pretrained_' if p else 'no_pretrained_')
                        )

                        if f == '1000x300' and d in ['kitti_voc', 'kitti_voc_small'] or f == '300x300':
                            runs.append(Run(name=experiment_name, dataset=d, format=f, random_aug=r, batch_norm=b, pretrained=p))
    return runs



if __name__ == "__main__":
    """
    define your runs here
    """

    #datasets = ['kitti_voc', 'VOC', 'toy_data', 'deformation_data', 'rotation_data', 'scale_data', 'translation_data']
    #datasets = ['deformation_data', 'rotation_data', 'scale_data', 'translation_data']
    datasets =  ['kitti_voc']
    formats = ['300x300']
    random_augs = [True]
    batch_norms = [True]
    pretrained = [True]
    runs = build_run_parameter_grid(datasets=datasets, formats=formats, random_augs=random_augs, batch_norm = batch_norms, pretrained=pretrained)


    apply_runs(runs)

    print("started " + str(len(runs)) + " runs")
