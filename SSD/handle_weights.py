from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from scattering_ssd import build_scattering_ssd
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
from ssd import build_ssd
from collections import OrderedDict

module_path = os.path.abspath(os.path.join('..'))
scattering_path = os.path.abspath('/home/marius/Desktop/Comp.Sci_Bachelor_2018/scattering_transform')
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append(scattering_path)


cfg = VOC_scat
dataset = VOCDetection(root=VOC_ROOT,
                       transform=SSDAugmentation(size_x=cfg['dim_x'],
                                                 size_y=cfg['dim_y'],
                                                 mean=VOC_MEANS,
                                                 random=True,
                                                 normalize=False,
                                                 sub_mean=True))


def adapt_names(weights, weights_namelist, filename, l2l_location):
    #give weights as in example:
    #vgg_weights = torch.load('weights/vgg16_bn-6c64b313.pth')
    new_weights = OrderedDict([])
    for name, param in weights.items():
        if 'features' in name:
            new_name = name.replace("features.", "")
            new_weights.update({new_name: param})

    last_2_layers = torch.load(l2l_location)
    for name, param in last_2_layers.items():
        new_weights.update({name: param})

    torch.save(new_weights, filename)


#get the last two layers from the already existing vgg16_reducedfc pretrained network
def extract_last_2_layers(list_of_new_names, filename):
    weights_location = 'weights/vgg16_reducedfc.pth'
    vgg_weights = torch.load(weights_location)
    new_weights = OrderedDict([])
    list_of_layer_2_names = ['31.weight', '31.bias', '33.weight', '33.bias']
    i = 0
    for name, param in vgg_weights.items():
        if name in list_of_layer_2_names:
            new_weights.update({list_of_new_names[i] : param})
            i += 1

    torch.save(new_weights, filename)

#extract_last_2_layers(['44.weight', '44.bias', '46.weight', '46.bias'], "weights/last_2_layers_bn.pth")

def prepare_pretrained_for_scattering(list_of_old_names, list_of_new_names, filename):
#weights_name_list is a list of the old names that are supposed to become the new ones
    assert(len(list_of_new_names) == len(list_of_old_names))
    weights_location = 'weights/vgg16_reducedfc.pth'
    vgg_weights = torch.load(weights_location)
    new_weights = OrderedDict([])
    i = 0
    for name, param in vgg_weights.items():
        if name in list_of_old_names:
            new_weights.update({list_of_new_names[i] : param})
            i += 1

    torch.save(new_weights, filename)

def prepare_pretrained_for_scattering_bn(list_of_old_names, list_of_new_names, filename):
#weights_name_list is a list of the old names that are supposed to become the new ones
    assert(len(list_of_new_names) == len(list_of_old_names))
    weights_location = 'weights/vgg16_reduced_bn.pth'
    vgg_weights = torch.load(weights_location)
    new_weights = OrderedDict([])
    i = 0
    for name, param in vgg_weights.items():
        if name in list_of_old_names:
            new_weights.update({list_of_new_names[i] : param})
            i += 1

    torch.save(new_weights, filename)

def prepare_pretrained_for_parallel_scattering():

    weights_location = 'weights/vgg16_reduced_bn.pth'
    vgg_weights = torch.load(weights_location)
    for name, param in vgg_weights.items():
         print(name, param.size())

    weight_tensor1 = torch.Tensor(128, 91, 3, 3)
    weight_tensor1 = nn.init.xavier_normal_(weight_tensor1)
    weight_tensor2 = torch.Tensor(256, 371, 3, 3)
    weight_tensor2 = nn.init.xavier_normal_(weight_tensor2)

    vgg_weights['7.weight'] = weight_tensor1
    vgg_weights['14.weight'] = weight_tensor2
    #vgg_weights['5.weight'] = weight_tensor1
    #vgg_weights['10.weight'] = weight_tensor2

    for name, param in vgg_weights.items():
         print(name, param.size())

    filename = 'weights/vgg16_reduced_bn_scat_par.pth'
    print("save the weights at: ", filename)
    torch.save(vgg_weights, filename)

def remove_transformation_layer(weights):

    vgg_weights = torch.load(weights, map_location='cpu')
    new_weights = OrderedDict([])
    for name, param in vgg_weights.items():
        if name not in ['transform_layer.weight', 'transform_layer.bias']:
            print(name, param.size())
            new_weights.update({name: param})

    print("save the weights at: ", weights)
    torch.save(new_weights, weights)



remove_transformation_layer('weights/scattering_parallel_ssd_J2_VOC_random_batch_norm_pretrained_13_125000.pth')

#prepare_pretrained_for_parallel_scattering()

#names for the scattering setup with J=2, this means vgg setup is [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M',  512, 512, 512] #removed 64, 64, 'M' for the scattering implementation, removed 'M' after 128

list_of_old_names_J2 = ['5.weight',
'5.bias',
'7.weight',
'7.bias',
'10.weight',
'10.bias',
'12.weight',
'12.bias',
'14.weight',
'14.bias',
'17.weight',
'17.bias',
'19.weight',
'19.bias',
'21.weight',
'21.bias',
'24.weight',
'24.bias',
'26.weight',
'26.bias',
'28.weight',
'28.bias',
'31.weight',
'31.bias',
'33.weight',
'33.bias']

list_of_new_names_J2 = ['0.weight',
 '0.bias',
 '2.weight',
 '2.bias',
 '4.weight',
 '4.bias',
 '6.weight',
 '6.bias',
 '8.weight',
 '8.bias',
 '11.weight',
 '11.bias',
 '13.weight',
 '13.bias',
 '15.weight',
 '15.bias',
 '18.weight',
 '18.bias',
 '20.weight',
 '20.bias',
 '22.weight',
 '22.bias',
 '25.weight',
 '25.bias',
 '27.weight',
 '27.bias']

#for scattering with batch norm
list_of_old_names_J2_bn = [
'7.weight',
'7.bias',
'8.weight',
'8.bias',
'8.running_mean',
'8.running_var',
'10.weight',
'10.bias',
'11.weight',
'11.bias',
'11.running_mean',
'11.running_var',
'14.weight',
'14.bias',
'15.weight',
'15.bias',
'15.running_mean',
'15.running_var',
'17.weight',
'17.bias',
'18.weight',
'18.bias',
'18.running_mean',
'18.running_var',
'20.weight',
'20.bias',
'21.weight',
'21.bias',
'21.running_mean',
'21.running_var',
'24.weight',
'24.bias',
'25.weight',
'25.bias',
'25.running_mean',
'25.running_var',
'27.weight',
'27.bias',
'28.weight',
'28.bias',
'28.running_mean',
'28.running_var',
'30.weight',
'30.bias',
'31.weight',
'31.bias',
'31.running_mean',
'31.running_var',
'34.weight',
'34.bias',
'35.weight',
'35.bias',
'35.running_mean',
'35.running_var',
'37.weight',
'37.bias',
'38.weight',
'38.bias',
'38.running_mean',
'38.running_var',
'40.weight',
'40.bias',
'41.weight',
'41.bias',
'41.running_mean',
'41.running_var',
'44.weight',
'44.bias',
'46.weight',
'46.bias']

list_of_new_names_J2_bn = [
'0.weight',
'0.bias',
'1.weight',
'1.bias',
'1.running_mean',
'1.running_var',
'3.weight',
'3.bias',
'4.weight',
'4.bias',
'4.running_mean',
'4.running_var',
'6.weight',
'6.bias',
'7.weight',
'7.bias',
'7.running_mean',
'7.running_var',
'9.weight',
'9.bias',
'10.weight',
'10.bias',
'10.running_mean',
'10.running_var',
'12.weight',
'12.bias',
'13.weight',
'13.bias',
'13.running_mean',
'13.running_var',
'16.weight',
'16.bias',
'17.weight',
'17.bias',
'17.running_mean',
'17.running_var',
'19.weight',
'19.bias',
'20.weight',
'20.bias',
'20.running_mean',
'20.running_var',
'22.weight',
'22.bias',
'23.weight',
'23.bias',
'23.running_mean',
'23.running_var',
'26.weight',
'26.bias',
'27.weight',
'27.bias',
'27.running_mean',
'27.running_var',
'29.weight',
'29.bias',
'30.weight',
'30.bias',
'30.running_mean',
'30.running_var',
'32.weight',
'32.bias',
'33.weight',
'33.bias',
'33.running_mean',
'33.running_var',
'36.weight',
'36.bias',
'38.weight',
'38.bias']


#prepare_pretrained_for_scattering(list_of_old_names_J2, list_of_new_names_J2, filename="weights/vgg16_pretrained_scat_J2.pth")
#prepare_pretrained_for_scattering_bn(list_of_old_names_J2_bn, list_of_new_names_J2_bn, filename="weights/vgg16_pretrained_scat_J2_bn.pth")

#test if it worked:
#inp_channels = 64

#ssd_net = build_scattering_ssd(phase='train', inp_channels=inp_channels, size_x=cfg['dim_x'], size_y=cfg['dim_y'], num_classes=cfg['num_classes'],
#                               cfg=cfg, batch_norm=True, pretrained=True)
#print(ssd_net.vgg)
# for name, param in ssd_net.vgg.items():
#     print(name, param.size())
#vgg16_pretrained_scat_J2_bn = torch.load('weights/vgg16_pretrained_scat_J2_bn.pth')
#print('Loading base network...')
#ssd_net.vgg.load_state_dict(vgg16_pretrained_scat_J2_bn)

scattering_J2_names = ['vgg.0.weight',
 'vgg.0.bias',
 'vgg.2.weight',
 'vgg.2.bias',
 'vgg.4.weight',
 'vgg.4.bias',
 'vgg.6.weight',
 'vgg.6.bias',
 'vgg.8.weight',
 'vgg.8.bias',
 'vgg.11.weight',
 'vgg.11.bias',
 'vgg.13.weight',
 'vgg.13.bias',
 'vgg.15.weight',
 'vgg.15.bias',
 'vgg.18.weight',
 'vgg.18.bias',
 'vgg.20.weight',
 'vgg.20.bias',
 'vgg.22.weight',
 'vgg.22.bias',
 'vgg.25.weight',
 'vgg.25.bias',
 'vgg.27.weight',
 'vgg.27.bias',
 'L2Norm.weight',
 'extras.0.weight',
 'extras.0.bias',
 'extras.1.weight',
 'extras.1.bias',
 'extras.2.weight',
 'extras.2.bias',
 'extras.3.weight',
 'extras.3.bias',
 'extras.4.weight',
 'extras.4.bias',
 'extras.5.weight',
 'extras.5.bias',
 'extras.6.weight',
 'extras.6.bias',
 'extras.7.weight',
 'extras.7.bias',
 'loc.0.weight',
 'loc.0.bias',
 'loc.1.weight',
 'loc.1.bias',
 'loc.2.weight',
 'loc.2.bias',
 'loc.3.weight',
 'loc.3.bias',
 'loc.4.weight',
 'loc.4.bias',
 'loc.5.weight',
 'loc.5.bias',
 'conf.0.weight',
 'conf.0.bias',
 'conf.1.weight',
 'conf.1.bias',
 'conf.2.weight',
 'conf.2.bias',
 'conf.3.weight',
 'conf.3.bias',
 'conf.4.weight',
 'conf.4.bias',
 'conf.5.weight',
 'conf.5.bias'
 ]

 #names for the weights of vgg16 with batch norm:

vgg16_bn_names = ['features.0.weight',
'features.0.bias',
'features.1.weight',
'features.1.bias',
'features.1.running_mean',
'features.1.running_var',
'features.3.weight',
'features.3.bias',
'features.4.weight',
'features.4.bias',
'features.4.running_mean',
'features.4.running_var',
'features.7.weight',
'features.7.bias',
'features.8.weight',
'features.8.bias',
'features.8.running_mean',
'features.8.running_var',
'features.10.weight',
'features.10.bias',
'features.11.weight',
'features.11.bias',
'features.11.running_mean',
'features.11.running_var',
'features.14.weight',
'features.14.bias',
'features.15.weight',
'features.15.bias',
'features.15.running_mean',
'features.15.running_var',
'features.17.weight',
'features.17.bias',
'features.18.weight',
'features.18.bias',
'features.18.running_mean',
'features.18.running_var',
'features.20.weight',
'features.20.bias',
'features.21.weight',
'features.21.bias',
'features.21.running_mean',
'features.21.running_var',
'features.24.weight',
'features.24.bias',
'features.25.weight',
'features.25.bias',
'features.25.running_mean',
'features.25.running_var',
'features.27.weight',
'features.27.bias',
'features.28.weight',
'features.28.bias',
'features.28.running_mean',
'features.28.running_var',
'features.30.weight',
'features.30.bias',
'features.31.weight',
'features.31.bias',
'features.31.running_mean',
'features.31.running_var',
'features.34.weight',
'features.34.bias',
'features.35.weight',
'features.35.bias',
'features.35.running_mean',
'features.35.running_var',
'features.37.weight',
'features.37.bias',
'features.38.weight',
'features.38.bias',
'features.38.running_mean',
'features.38.running_var',
'features.40.weight',
'features.40.bias',
'features.41.weight',
'features.41.bias',
'features.41.running_mean',
'features.41.running_var',
'classifier.0.weight',
'classifier.0.bias',
'classifier.3.weight',
'classifier.3.bias',
'classifier.6.weight',
'classifier.6.bias'
]

#vgg16_bn_weights =  torch.load('weights/vgg16_bn-6c64b313.pth')
#adapt_names(weights=vgg16_bn_weights, weights_namelist=vgg16_bn_names, filename='weights/vgg16_reduced_bn.pth', l2l_location='weights/last_2_layers_bn.pth')

#print("vgg16_reduced_bn names: ")
#vgg16_reduced_bn = torch.load('weights/vgg16_reduced_bn.pth')
#for name, param in vgg16_reduced_bn.items():
#    print(name, param.size())

#test whether this is loadable:
# vgg16_reduced_bn = torch.load('weights/vgg16_reduced_bn.pth')
# print('Loading base network...')
# ssd_net.vgg.load_state_dict(vgg16_reduced_bn)
#print('weights: ', ssd_net.vgg.state_dict())

# vgg16_reducedfc weights names:

vgg16_reducedfc_names = [
'0.weight',
'0.bias',
'2.weight',
'2.bias',
'5.weight',
'5.bias',
'7.weight',
'7.bias',
'10.weight',
'10.bias',
'12.weight',
'12.bias',
'14.weight',
'14.bias',
'17.weight',
'17.bias',
'19.weight',
'19.bias',
'21.weight',
'21.bias',
'24.weight',
'24.bias',
'26.weight',
'26.bias',
'28.weight',
'28.bias',
'31.weight',
'31.bias',
'33.weight',
'33.bias',
]


vgg16_reducedfc_bn_names = [
'0.weight',
'0.bias',
'1.weight',
'1.bias',
'1.running_mean',
'1.running_var',
'3.weight',
'3.bias',
'4.weight',
'4.bias',
'4.running_mean',
'4.running_var',
'7.weight',
'7.bias',
'8.weight',
'8.bias',
'8.running_mean',
'8.running_var',
'10.weight',
'10.bias',
'11.weight',
'11.bias',
'11.running_mean',
'11.running_var',
'14.weight',
'14.bias',
'15.weight',
'15.bias',
'15.running_mean',
'15.running_var',
'17.weight',
'17.bias',
'18.weight',
'18.bias',
'18.running_mean',
'18.running_var',
'20.weight',
'20.bias',
'21.weight',
'21.bias',
'21.running_mean',
'21.running_var',
'24.weight',
'24.bias',
'25.weight',
'25.bias',
'25.running_mean',
'25.running_var',
'27.weight',
'27.bias',
'28.weight',
'28.bias',
'28.running_mean',
'28.running_var',
'30.weight',
'30.bias',
'31.weight',
'31.bias',
'31.running_mean',
'31.running_var',
'34.weight',
'34.bias',
'35.weight',
'35.bias',
'35.running_mean',
'35.running_var',
'37.weight',
'37.bias',
'38.weight',
'38.bias',
'38.running_mean',
'38.running_var',
'40.weight',
'40.bias',
'41.weight',
'41.bias',
'41.running_mean',
'41.running_var',
'44.weight',
'44.bias',
'46.weight',
'46.bias'
]
