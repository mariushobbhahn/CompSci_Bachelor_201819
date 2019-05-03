#!/usr/bin/python
import os
import sys
from string import Template

BATCHFILE_TEMPLATE = "batchTemplate.tmp"
RUNFILE_TEMPLATE = "runTemplate.tmp"
FILEDIRECTORY = "BatchFiles/"
WORKPATH = os.path.dirname(os.path.realpath(sys.argv[0]))[11:]  # 11 to remove /mnt/beegfs
print("workpath: ", WORKPATH)

# ADAPT:
# NETFILENAME = "train.py"


# Optimizer code:
# Admam : AD
# Gradient descent with Momentum: GM
# Parabola Approximation Optimizer: PA

class Run:
    """
    defines hyper parameters for one network
    """

    def __init__(self, name, dataset, random_aug, normalize, sub_mean):
        self.experiment_name = name
        self.dataset = str(dataset)
        self.random_aug = random_aug
        self.normalize = normalize
        self.sub_mean = sub_mean

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


def build_run_parameter_grid(datasets, random_augs, normalize, sub_mean):
    runs = []
    for d in datasets:
        for r in random_augs:
            for n in normalize:
                for s in sub_mean:
                    experiment_name = str('ssd_' +
                        str(d) + '_' +
                        '{}'.format('random_' if r else 'no_random_') +
                        '{}'.format('sub_mean_' if s else 'no_sub_mean_') +
                        '{}'.format('normalize_' if n else 'no_normalize_')
                    )

                    runs.append(Run(name=experiment_name, dataset=d, random_aug=r, normalize=n, sub_mean=s))
    return runs


def build_run_parameter_grid2(param_list, name):
    runs = []
    __recursive_build_run_parameter_grid2(param_list, 0, [], name, runs)
    return runs


def __recursive_build_run_parameter_grid2(param_list, i, runpars, name, runs):
    if i < len(param_list):
        for a in param_list[i]:
            name1 = name + "_" + str(a)
            rp = runpars[:]
            rp.append(a)
            __recursive_build_run_parameter_grid2(param_list, i + 1, rp, name1, runs)
    else:
        runs.append(Run(name, *runpars))
        print(name)


if __name__ == "__main__":
    """
    define your runs here
    """

    datasets = ['VOC', 'kitti_voc', 'kitti_voc_small']
    random_augs = [True, False]
    normalize = [True, False]
    sub_means = [True, False]
    runs = build_run_parameter_grid(datasets=datasets, random_augs=random_augs, normalize=normalize, sub_mean=sub_means)
    # PA:   experiment_name, random_seed, optimizer, train_data_size, train_time, batchsize, measuring_step_size, momentum, loose_approximation_factor, max_stepsize, decay, additional

    # Adam: experiment_name,random_seed,train_data_size,timeinmin,batchsize, learning rate, beta1,beta2,epsilon, decayrate, decaysteps,
    # MG: experiment_name,random_seed,train_data_size,timeinmin,batchsize, learning rate, momentum,-,-,decayrate, decaysteps, )

    # runs=build_run_parameter_grid("AD",[1337],["AD"],[45000],[100],[100],[0.00001,0.0001,0.001,0.01,0.1],[0.9,0.8],[0.999,0.9],[1e-08,0.1,1])
    # runs=build_run_parameter_grid("GM",[1337],["GM"],[45000],[100],[100],[0.001,0.005,0.01,0.05,0.1],[0.9,0.8,0.6,0.4,0.2,0],[0],[0])

    # runs=build_run_parameter_grid("GM",[1337],["GM","GMP","GMEX","GMCOS"],[45000],[100],[100],[0.001,0.01,0.1],[0.9,0.6],[0],[0],[0.995],[0.99])

    #runs1 = build_run_parameter_grid2(
    #    [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0], [1], [1,0.1], [1], [0]], "Pa_resnet_base") #6*10
    # runs2 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0.2, 0.4, 0.6, 0.8, 1], [1], [1,0.1], [1], [0]], "Pa_resnet_mom") # 30 *10
    # runs3 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0], [0.2, 0.4, 0.6, 0.8,1.2], [1,0.1], [1], [0]], "Pa_resnet_lo_ap")# 30*10
    # runs4 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0], [1], [1,0.1], [0.8,0.85,0.9,0.95], [0]], "Decay")# 24*10
    # runs5 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0.2, 0.4, 0.6, 0.8, 1], [0.2, 0.4, 0.6, 0.8,1.2], [1,0.1], [1], [0]], "Pa_resnet_mom_and_lo_ap") # 150 *10
    # runs6 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0.2, 0.4, 0.6, 0.8, 1], [1], [1,0.1], [0.8,0.85,0.9,0.95], [0]], "Pa_resnet_mom_and_decay") # 120 *10
    # runs7 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0], [0.2, 0.4, 0.6, 0.8,1.2], [1,0.1], [0.8,0.85,0.9,0.95], [0]], "Pa_resnet_lo_ap_and_decay")# 120*10
    # runs8 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0.2, 0.4, 0.6, 0.8,1.2], [0.2, 0.4, 0.6, 0.8,1.2], [1,0.1], [0.8,0.85,0.9,0.95], [0]], "Pa_resnet_mom_and_lo_ap_and_decay")#600*10
    # runs8 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01,0.001], [0.2, 0.4, 0.6, 0.8,1], [0.2, 0.4, 0.6, 0.8,1.2], [1,0.1], [0.8,0.85,0.9,0.95], [0]], "Pa_resnet_mom_and_lo_ap_and_decay")#600*10

    # # whole grid search 600*10 networks

    # good params
    # runs1 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01], [0,0.2, 0.4, 0.6], [0.2, 0.4, 0.6], [1], [0.8,0.85,0.9], [0]], "Pa_vgg_net")#2*3*4*3*10 -> 720
    #
    #
    # runs1 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["GMEX"], [45000], [20], [100], [0.1,0.01,0.001,0.0001], [0,0.8,0.9,0.95,1], [0], [0], [0.8,0.85,0.9,0.95,1],[450]], "resnet_GM") #1000
    #
    # runs1 = build_run_parameter_grid2(
    #     [[1,2,3,4,5,6,7,8,9,10], ["ADEX"], [45000], [20], [100], [0.1,0.01,0.001,0.0001], [0.9], [0.999], [1e-8,1,0.1],[0.8,0.85,0.9,0.95,1],[450]], "resnet_Adam") #600
    #
    # runs.extend(runs1)
  #  runs.extend(runs2)
   # runs.extend(runs3)

    apply_runs(runs)

    print("started " + str(len(runs)) + " runs")

#5179 failed
