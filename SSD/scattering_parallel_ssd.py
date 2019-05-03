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



class Scattering2dParallelSSD(nn.Module):
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

    def __init__(self, inp_channels, phase, size_x, size_y, base, extras, head, num_classes, cfg, pretrained_weights):
        super(Scattering2dParallelSSD, self).__init__()
        self.phase = phase
        self.K = inp_channels
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size_x = size_x
        self.size_y = size_y

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


    def forward(self, x, scatterings, Ks):
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
        # if self.pretrained_weights:
        #     x = x.view(x.size(0), 51, 75, 75)
        #     x = self.transform_layer(x)
        #     self.K = 64 #this is only a quick fix; refactor pls
        # else:
        #     x = x.view(x.size(0), self.K, 75, 75)

        #scattering0_images = scatterings[0](x.cuda())
        scattering1_images = scatterings[0](x.cuda())
        scattering1_images = scattering1_images.view(scattering1_images.size(0), scattering1_images.size(1) * scattering1_images.size(2), 150, 150)
        #print("scattering1 images size: ", scattering1_images.size())
        scattering2_images = scatterings[1](x.cuda())
        scattering2_images = scattering2_images.view(scattering2_images.size(0), scattering2_images.size(1) * scattering2_images.size(2), 75, 75)


        #x = torch.cat((x, scattering0_images), dim=1)
        x = self.vgg[0](x) #64
        x = self.vgg[1](x) #relu
        x = self.vgg[2](x) #64
        x = self.vgg[3](x) #relu
        x = self.vgg[4](x) #M
        #print("x size before first scattering: ", x.size())
        x = torch.cat((x, scattering1_images), dim=1)
        x = self.vgg[5](x) #128
        x = self.vgg[6](x) #relu
        x = self.vgg[7](x) #128
        x = self.vgg[8](x) #relu
        x = self.vgg[9](x) #M
        x = torch.cat((x, scattering2_images), dim=1)
        x = self.vgg[10](x) #256
        x = self.vgg[11](x) #relu
        x = self.vgg[12](x) #256
        x = self.vgg[13](x) #relu
        x = self.vgg[14](x) #256
        x = self.vgg[15](x) #ReLU
        x = self.vgg[16](x) #C
        x = self.vgg[17](x) #512
        x = self.vgg[18](x) #ReLU
        x = self.vgg[19](x) #512
        x = self.vgg[20](x) #ReLU
        x = self.vgg[21](x) #512
        x = self.vgg[22](x) #ReLU

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
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


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    #Ks is the number of addiditional filters that are added by a given set
    layers = []
    in_channels = i

    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_pool_ceil = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    relu = nn.ReLU(inplace=True)
    #
    conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #scattering1 is added here, therefore the 91=64 + 27
    layers += [conv1_1, relu, conv1_2, relu, max_pool]
    conv2_1 = nn.Conv2d(91, 128, kernel_size=3, padding=1)
    conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    layers += [conv2_1, relu, conv2_2, relu, max_pool]
    #scattering2 is added here, therefore the 179 = 128 + 51
    conv3_1 = nn.Conv2d(179, 256, kernel_size=3, padding=1)
    conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    layers += [conv3_1, relu, conv3_2, relu, conv3_3, relu, max_pool_ceil]
    #
    conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    layers += [conv4_1, relu, conv4_2, relu, conv4_3, relu, max_pool]
    #
    conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    layers += [conv5_1, relu, conv5_2, relu, conv5_3, relu]

    """
    for v in cfg:
        #print("layers: ", layers)
        if v == 'M':
            #print("v==M")
            #print("v: ", v, "type: ", type(v))
            #print("rep: ", repr(v))
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            scat_i += 1
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
    """

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


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]       #corresponds to conv4_3 and conv7

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


def build_scattering_parallel_ssd(phase, inp_channels, size_x=300, size_y=300, num_classes=21, cfg='voc', pretrained=False, batch_norm=False):
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
                                     cfg['mbox'], num_classes)
    return Scattering2dParallelSSD(phase=phase, inp_channels=inp_channels,
                           size_x=size_x, size_y=size_y, base=base_, extras=extras_,
                           head=head_, num_classes=num_classes, cfg=cfg, pretrained_weights=pretrained)
