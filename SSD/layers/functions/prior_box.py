from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.size_x = cfg['dim_x']
        self.size_y = cfg['dim_y']
        self.min_size = min(self.size_x, self.size_y)
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps_y = cfg['feature_maps_y']
        self.feature_maps_x = cfg['feature_maps_x']
        self.s_sizes = cfg['s_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        #print("ar during prior box: ", self.aspect_ratios)
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []                                                     #all locations have the form [center_x, center_y, width, height]
        for k in range(len(self.feature_maps_y)):                     #iterate over the length feature maps y
            for i, j in product(range(self.feature_maps_x[k]), range(self.feature_maps_y[k])):     #iterate over the cross product of the feature maps x and y
                f_k = self.min_size / self.steps[k]               #?
                # unit center x,y
                cx = (i + 0.5) / f_k                                 #give the x-position of the anchor box
                cy = (j + 0.5) / f_k                                 #give the y-position of the anchor box

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.s_sizes[k]/self.size_y                 #give relative size of box in x direction   #Q: do we need size.y if we want quadratic boxes?
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.s_sizes[k+1]/self.min_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    #mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]   #uncomment this line to add the same bounding box for
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
