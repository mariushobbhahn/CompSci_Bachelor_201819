import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import kitti #, voc, coco
import sys
#sys.path.append('/home/hobbhahn/scattering_transform')
#from scattering import Scattering2D
import os



class Scattering2dSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Note: additionally the class has been adapted to use the scattering transform

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers

    """

    def __init__(self, inp_channels, phase, size_x, size_y, base, extras, head, num_classes, cfg, batch_norm, pretrained_weights):
        super(Scattering2dSSD, self).__init__()
        self.phase = phase
        self.K = inp_channels
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size_x = size_x
        self.size_y = size_y
        self.batch_norm = batch_norm

        # SSD network
        self.pretrained_weights = pretrained_weights
        if pretrained_weights:
            self.transform_layer = nn.Conv2d(51, 64, kernel_size=1, padding=0)

        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45, cfg)


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        #transform scattering input to correct format:
        #print("self.K: ", self.K)
        #print("x.size: ", x.size())
        if self.pretrained_weights:
            x = x.view(x.size(0), 51, 75, 75)
            x = self.transform_layer(x)
            self.K = 64 #this is only a quick fix; refactor pls 
        else:
            x = x.view(x.size(0), self.K, 75, 75)

        # apply vgg up to conv4_3 relu
        for k in range(17 + 8 if self.batch_norm else 17):                 #changed 23 to 18 since we removed 64, relu, 64, relu, 'M' as the first 5 layers, -1 for removing 'M'
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(17 + 8 if self.batch_norm else 17, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            #print("size: ", x.size())
            #print("loc.size(): ", loc.size())
            #print(" self.priors.type(type(x.data)): ",  self.priors.type(type(x.data)))
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    #only loads those layers that are needed in current system
    def load_my_state_dict(self, state_dict):

            own_state = self.state_dict()
            print("own_state: ", own_state)
            for name, param in state_dict.items():
                print("name: ", name)
                if name not in own_state:
                     continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    print("param: ", param)
                    param = param.data
                own_state[name].copy_(param)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

"""
def add_scattering(cfg, i, mode=1, batch_norm=False):
    #first layers replaced for scattering transform
    layers = []
    in_channels = i     # actually I dont think we are using this at all for the scattering
    for v in cfg:
        #if v == 'M':
        #    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #else:
        if mode == 1:
            scattering = Scattering2D(M=300, N=300, J=v, order2=False, pre_pad=True) #pre_pad true such that we can have an integer number of filters
            K = (1 + 10*8)*3          #for J = 10 and L = 8 per default
        else:
            scattering = Scattering2D(M=300, N=300, J=v, pre_pad=False)
            K = (1 + 10*8 + 5*9*64)*3
        layers += [scattering]

    return(layers, K)
"""

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i

    #print("time during vgg: ", datetime.now().time())
    #print("in_channels: ", in_channels, "type: ", type(in_channels))

    for v in cfg:
        #print("layers: ", layers)
        if v == 'M':
            #print("v==M")
            #print("v: ", v, "type: ", type(v))
            #print("rep: ", repr(v))
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            #print("else")
            #print("in_channels: ", in_channels, "type: ", type(in_channels))
            #print("v: ", v, "type: ", type(v))
            #print("rep: ", repr(v))
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, batch_norm):
    loc_layers = []
    conf_layers = []
    vgg_source = [15 + 7 if batch_norm else 15, -2]       #corresponds to conv4_3 and conv7
    
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

"""
scattering = {                                                          #have one layer of scattering coefficients with J=10
    '300': [10]
}
"""

#base = {
#    '300': [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M',  #removed 64, 64, 'M' for the scattering implementation, removed 'M' after 128
#            512, 512, 512],
#    '512': [],
#}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [8, 8, 8, 8, 8, 8],  # number of boxes per feature map location
    '512': [],
}


def build_scattering_ssd(phase, inp_channels, size_x=300, size_y=300, num_classes=21, cfg='voc', pretrained=False, batch_norm=False):
    print("inp_channels: ", inp_channels)
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size_y != 300:
        print("ERROR: You specified size " + repr(size_y) + ". However, " +
              "currently only SSD300 (size_y=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(cfg['base'], i=inp_channels, batch_norm=batch_norm),          #81 is for mode = 1 Calculation is explained in paper and code
                                     add_extras(cfg=extras[str(size_y)], i=1024),
                                     cfg['mbox'], num_classes, batch_norm)
    return Scattering2dSSD(phase=phase, inp_channels=inp_channels,
                           size_x=size_x, size_y=size_y, base=base_, extras=extras_,
                           head=head_, num_classes=num_classes, cfg=cfg, pretrained_weights=pretrained, batch_norm=batch_norm)
