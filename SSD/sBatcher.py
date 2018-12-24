#!/usr/bin/python
import os
import sys
from string import Template

BATCHFILE_TEMPLATE = "batchTemplate.tmp"
RUNFILE_TEMPLATE = "runTemplate.tmp"
FILEDIRECTORY = "BatchFiles/"
WORKPATH = os.path.dirname(os.path.realpath(sys.argv[0]))[11:]  # 11 to remove /mnt/beegfs

# ADAPT:
NETFILENAME = "code_/main.py"


# Optimizer code:
# Admam : AD
# Gradient descent with Momentum: GM
# Parabola Approximation Optimizer: PA

class Run:
    """
    defines hyper parameters for one network
    """

    def __init__(self, experiment_name, random_seed, optimizer, train_data_size, train_time, batchsize,
                 measuring_step_size, momentum, loose_approximation_factor, max_stepsize, decay, additional):
        self.random_seed = str(random_seed)
        self.train_data_size = str(train_data_size)
        self.experiment_name = str(experiment_name)
        self.train_time = str(train_time)
        self.batchsize = str(batchsize)
        self.measuring_step_size = str(measuring_step_size)
        self.momentum = str(momentum)
        self.optimizer = str(optimizer)
        self.loose_approximation_factor = loose_approximation_factor
        self.max_stepsize = max_stepsize
        self.decay = decay
        self.additional = additional


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
        batchcontent = t.substitute(runfile=runFilePath, experiment_name=run.experiment_name, workpath=WORKPATH,
                                    netfile=NETFILENAME)
        batch_file_path = FILEDIRECTORY + run.experiment_name + "_" + str(i) + ".sbatch"
        with open(batch_file_path, "w+") as batchfile:
            batchfile.write(batchcontent)
        os.system("chmod 775 " + batch_file_path)
        os.system("sbatch " + batch_file_path)


def build_run_parameter_grid(basename, random_seeds, optimizers, train_data_sizees, train_times, batchsizes,
                             measuring_step_sizees, momentums, loose_approximation_factors, max_stepsizes, decays, additional):
    runs = []
    for r in random_seeds:
        for o in optimizers:
            for t in train_data_sizees:
                for e in train_times:
                    for b in batchsizes:
                        for l in measuring_step_sizees:
                            for m in momentums:
                                for ra in loose_approximation_factors:
                                    for ma in max_stepsizes:
                                        for maD in decays:
                                            for msD in additional:
                                                name = basename
                                                if len(random_seeds) > 1:
                                                    name += "_" + str(r)
                                                if len(optimizers) > 1:
                                                    name += "_" + str(o)
                                                if len(train_data_sizees) > 1:
                                                    name += "_" + str(t)
                                                if len(train_times) > 1:
                                                    name += "_" + str(e)
                                                if len(batchsizes) > 1:
                                                    name += "_" + str(b)
                                                if len(measuring_step_sizees) > 1:
                                                    name += "_" + str(l)
                                                if len(momentums) > 1:
                                                    name += "_" + str(m)
                                                if len(loose_approximation_factors) > 1:
                                                    name += "_" + str(ra)
                                                if len(max_stepsizes) > 1:
                                                    name += "_" + str(ma)
                                                if len(decays) > 1:
                                                    name += "_" + str(maD)
                                                if len(additional) > 1:
                                                    name += "_" + str(msD)
                                                runs.append(Run(name, r, o, t, e, b, l, m, ra, ma, maD, msD))
    return runs
    # def buildRunParameterGridRecursive(basename,paramlist):
    #     if len(paramlist) >0:
    #         if len(paramlist[0]) > 1:
    #             name += "_" + str(r):
    #      for param in paramlist[0]:


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
    runs = []
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
    runs1 = build_run_parameter_grid2(
        [[1,2,3,4,5,6,7,8,9,10], ["PA"], [45000], [20], [100], [0.1,0.01], [0,0.2, 0.4, 0.6], [0.2, 0.4, 0.6], [1], [0.8,0.85,0.9], [0]], "Pa_vgg_net")#2*3*4*3*10 -> 720


    runs1 = build_run_parameter_grid2(
        [[1,2,3,4,5,6,7,8,9,10], ["GMEX"], [45000], [20], [100], [0.1,0.01,0.001,0.0001], [0,0.8,0.9,0.95,1], [0], [0], [0.8,0.85,0.9,0.95,1],[450]], "resnet_GM") #1000

    runs1 = build_run_parameter_grid2(
        [[1,2,3,4,5,6,7,8,9,10], ["ADEX"], [45000], [20], [100], [0.1,0.01,0.001,0.0001], [0.9], [0.999], [1e-8,1,0.1],[0.8,0.85,0.9,0.95,1],[450]], "resnet_Adam") #600

    runs.extend(runs1)
  #  runs.extend(runs2)
   # runs.extend(runs3)

    apply_runs(runs)

    print("started " + str(len(runs)) + " runs")

#5179 failed