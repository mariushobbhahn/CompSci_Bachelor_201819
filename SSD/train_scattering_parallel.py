from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from scattering_parallel_ssd import build_scattering_parallel_ssd
import os
import sys
import time
from datetime import datetime
print("time at import: ", datetime.now().time())
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

sys.path.append('/home/hobbhahn/scattering_transform')
from scattering import Scattering2D


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='toy_data', choices=['VOC', 'kitti_voc', 'kitti_voc_small', 'toy_data'],
                    type=str, help='VOC, kitti_voc, toy_data or kitti_voc_small')
parser.add_argument('--config', default='300x300', choices=['300x300', '1000x300'],
                    type=str, help='size of the imagescales')
parser.add_argument('--random', default=True, type=str2bool,
                    help='use many different random augmentations. For details see utils/augmentations.py')
parser.add_argument('--subtract_mean', default=True, type=str2bool,
                    help='subtract the color means before training')
parser.add_argument('--normalize', default=False, type=str2bool,
                    help='normalize images before training')
parser.add_argument('--mode', default=1, type=int, help='number of sequential scattering transforms')
parser.add_argument('--J', default=2, type=int, help='J determines the size of the window scattering transform: height = 2J')
parser.add_argument('--gen', default='12.8', type=str, help='generation of tests')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
#parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
#                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='/home/hobbhahn/SSD/weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained_weights', default=False, help='use pretrained weights for VGG')
args = parser.parse_args()

WEIGHTS_NAME = str('scattering_parallel_ssd_' +
                    'J{}_'.format(args.J) +
                    '{}_'.format(args.dataset) +
                    '{}_'.format('pretrained' if args.pretrained_weights else 'no_pretrained') +
                    '{}_'.format('random' if args.random else 'no_random') +
                    '{}_'.format(args.gen)
                    )


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train(inp_channels, scattering, dataset, config):


    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_scattering_parallel_ssd(phase='train', inp_channels=inp_channels, size_x=cfg['dim_x'], size_y=cfg['dim_y'], num_classes=cfg['num_classes'], cfg=cfg, pretrained=args.pretrained_weights)


    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.pretrained_weights:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network from: {}'.format(args.save_folder + args.basenet))
        ssd_net.vgg.load_state_dict(vgg_weights)
    #else do nothing and train from scratch

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=args.momentum,           #replaced args.lr
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, cfg=cfg, use_gpu=args.cuda)

    print("time after building: ", datetime.now().time())

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    print("time after loading data: ", datetime.now().time())

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter'] + 100):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index, lr_in=cfg['lr'])

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)


        if args.cuda:
            #add scattering to the images
            #images = scattering(images.cuda())
            #print("images.size(): ", images.size())
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            #add scattering to the images
            #images = scattering(images)
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward

        # print("images.size(): ", images.size())

        t0 = time.time()
        out = net(images, scatterings, Ks)
        # backprop
        optimizer.zero_grad()
        # loc_data, conf_data, priors = out
        # print("loc_data: ", loc_data.size())
        # print("conf_data:", conf_data.size())
        # print("priors: ", priors.size())
        # print("target: ", targets)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % cfg['max_iter'] == 0:
            print('Saving state, iter, file:', iteration, WEIGHTS_NAME)
            torch.save(ssd_net.state_dict(), 'weights/' + WEIGHTS_NAME +
                       repr(iteration) + '.pth')


def adjust_learning_rate(optimizer, gamma, step, lr_in):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr_in * (gamma ** (step))                                  #replaced args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    print("time before calling train(): ", datetime.now().time())

    if args.dataset == 'VOC':
        cfg = voc_scat_parallel
        dataset = VOCDetection(root=VOC_ROOT,
                               transform=SSDAugmentation(size_x=cfg['dim_x'],
                                                         size_y=cfg['dim_y'],
                                                         mean=VOC_MEANS,
                                                         random=args.random,
                                                         normalize=args.normalize,
                                                         sub_mean=args.subtract_mean))

    elif args.dataset == 'toy_data':
        cfg = toy_data_scat_parallel
        dataset = toydataDetection(root=toy_data_ROOT,
                               transform=SSDAugmentation(size_x=cfg['dim_x'],
                                                         size_y=cfg['dim_y'],
                                                         mean=(0,0,0),
                                                         random=args.random,
                                                         normalize=args.normalize,
                                                         sub_mean=args.subtract_mean))


    elif args.dataset == 'kitti_voc':
        if args.config == '300x300':
            cfg = kitti300x300
        elif args.config == '1000x300':
            cfg = kitti1000x300
        else:
            raise ValueError('The given configuration is not possible')

        dataset = kittiDetection(root = kitti_ROOT,
                                transform = SSDAugmentation(size_x=cfg['dim_x'],
                                                            size_y=cfg['dim_y'],
                                                            mean=KITTI_MEANS,
                                                            random=args.random,
                                                            normalize=args.normalize,
                                                            sub_mean=args.subtract_mean))

    elif args.dataset == 'kitti_voc_small':
        if args.config == '300x300':
            cfg = kitti300x300
        elif args.config == '1000x300':
            cfg = kitti1000x300
        else:
            raise ValueError('The given configuration is not possible')

        dataset = kitti_small_Detection(root = kitti_small_ROOT,
                                transform = SSDAugmentation(size_x=cfg['dim_x'],
                                                            size_y=cfg['dim_y'],
                                                            mean=KITTI_MEANS,
                                                            random=args.random,
                                                            normalize=args.normalize,
                                                            sub_mean=args.subtract_mean))

    print("time after config: ", datetime.now().time())
    print("config: ", cfg)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #scattering0:  J=0, should be an input equivalent to the image size
    #J0 = 0
    #scattering0 = Scattering2D(M=cfg['dim_x'], N=cfg['dim_y'], J=J0, order2=False, pre_pad=False) #pre_pad true such that we can have an integer number of filters
    #K0 = (1 + J0*8)*3          #for J = 10 and L = 8 per default

    #scattering1: J=1, should be 150 by 150 since image size is reduced
    J1 = 1
    scattering1 = Scattering2D(M=cfg['dim_x'], N=cfg['dim_y'], J=J1, pre_pad=False, order2=False)
    K1 = (1 + J1*8)*3          #for J = 10 and L = 8 per default

    #scattering2: J=2, should be 75 by 75 since image size is reduced again
    J2 = 2
    scattering2 = Scattering2D(M=cfg['dim_x'], N=cfg['dim_y'], J=J2, pre_pad=False, order2=False)
    K2 = (1 + J2*8)*3          #for J = 10 and L = 8 per default

    #K = (1 + J1*8 + 0.5*J1*(J1-1)*64)*3
    if use_cuda:
        #scattering0 = scattering0.cuda()
        scattering1 = scattering1.cuda()
        scattering2 = scattering2.cuda()

    scatterings = [scattering1, scattering2] #scattering0,
    Ks = [K1, K2] #K0,

    train(scattering=scatterings, dataset=dataset, config=cfg, inp_channels=3)
