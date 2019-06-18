# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
  (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

VOC_MEANS =(104, 117, 123)    #(102.93, 111.36, 116.52)
KITTI_MEANS = (96.19429688, 95.55136972, 91.34692262)

# SSD300 CONFIGS
voc = {
'num_classes': 21,
'lr': 1e-3,
'lr_bn': 3e-3,
'lr_steps': (80000, 100000, 120000),
'max_iter': 125000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],
'variance': [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'VOC',
}

voc_small = {
'num_classes': 21,
'lr': 1e-3,
'lr_bn': 3e-3,
'lr_steps': (4000, 4500),
'max_iter': 5000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],
'variance': [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'VOC_small',
}



kitti300x300 = {
'num_classes': 5,
'lr': 1e-4,
'lr_bn': 1e-3,
'lr_steps': (100000, 110000, 120000),
'max_iter': 125000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
#'feature_maps_x': [125, 62, 31, 16, 14, 12],
'dim_x': 300,
#'dim_x': 1000,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
#'s_sizes': [10, 20, 45, 80, 130, 210, 270],       #for self-chosen values
's_sizes': [30, 60, 105, 150, 195, 240, 285],      #for s_min = 0.2 and s_max = 0.95
#'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
# 'aspect_ratios': [
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4]
# ],
'aspect_ratios': [
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3]
],
'mbox': [6, 6, 6, 6, 6, 6],
'variance' : [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip' : True,
'name': 'kitti300x300',
}

kitti1000x300 = {
'num_classes': 5,                                   # number of Classes
'lr': 1e-4,
'lr_bn': 1e-3,                                         # learning rate in the beginning
'lr_steps': (150000, 175000, 185000),
'max_iter': 200000,                                 # amounts of iterations
'feature_maps_y': [38, 19, 10, 5, 3, 1],            # number of feature maps in y direction
'feature_maps_x': [125, 62, 31, 16, 14, 12],        # number of feature maps in x direction
#'dim_x': 300,
'dim_x': 1000,                                      # x size of image after rescaling
'dim_y': 300,                                       # y size of image after rescaling
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [10, 20, 45, 80, 130, 210, 270],       #for self-chosen values
#'s_sizes': [30, 60, 105, 150, 195, 240, 285],      #for s_min = 0.2 and s_max = 0.95
#'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
# 'aspect_ratios': [
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4]
# ],
'aspect_ratios': [
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3]
],
'mbox': [6, 6, 6, 6, 6, 6],
'variance' : [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip' : True,
'name': 'kitti1000x300',
}


kitti300x300_small = {
'num_classes': 5,
'lr': 1e-4,
'lr_bn': 1e-3,
'max_iter': 50000,
'lr_steps': (40000, 45000),
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
#'feature_maps_x': [125, 62, 31, 16, 14, 12],
'dim_x': 300,
#'dim_x': 1000,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [10, 20, 45, 80, 130, 210, 270],       #for self-chosen values
#'s_sizes': [30, 60, 105, 150, 195, 240, 285],      #for s_min = 0.2 and s_max = 0.95
#'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
# 'aspect_ratios': [
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4]
# ],
'aspect_ratios': [
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3]
],
'mbox': [6, 6, 6, 6, 6, 6],
'variance' : [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip' : True,
'name': 'kitti300x300_small',
}

kitti1000x300_small = {
'num_classes': 5,                                   # number of Classes
'lr': 1e-4,
'lr_bn': 1e-3,                                         # learning rate in the beginning
'max_iter': 50000,
'lr_steps': (40000, 45000),                                # amounts of iterations
'feature_maps_y': [38, 19, 10, 5, 3, 1],            # number of feature maps in y direction
'feature_maps_x': [125, 62, 31, 16, 14, 12],        # number of feature maps in x direction
#'dim_x': 300,
'dim_x': 1000,                                      # x size of image after rescaling
'dim_y': 300,                                       # y size of image after rescaling
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [10, 20, 45, 80, 130, 210, 270],       #for self-chosen values
#'s_sizes': [30, 60, 105, 150, 195, 240, 285],      #for s_min = 0.2 and s_max = 0.95
#'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
# 'aspect_ratios': [
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4],
#     [0.25, 0.33, 0.5, 2, 3, 4]
# ],
'aspect_ratios': [
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3]
],
'mbox': [6, 6, 6, 6, 6, 6],
'variance' : [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip' : True,
'name': 'kitti1000x300',
}

toy_data = {
'num_classes': 4,
'lr': 1e-4,
'lr_bn': 1e-3,
#'lr_steps': (80000, 100000, 120000),
#'max_iter': 125000,
'lr_steps':(60000, 65000, 70000),
'max_iter':75000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 4, 4, 4, 4, 4],
'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'toy_data',
}

toy_data_scat = {
'num_classes': 4,
'lr': 3e-4,
'lr_bn': 3e-3,
'lr_steps': (80000, 100000, 120000),
'max_iter': 125000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 4, 4, 4, 4, 4],
'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
'clip': True,
'name': 'toy_data_scat',
}


toy_data_small_scat = {
'num_classes': 4,
'lr': 3e-4,
'lr_bn': 3e-3,
'lr_steps': (4000, 4500),
'max_iter': 5000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 4, 4, 4, 4, 4],
'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
'clip': True,
'name': 'toy_data_scat',
}



VOC_scat = {
'num_classes': 21,
'lr': 3e-4,
'lr_bn': 1e-3,
'lr_steps': (80000, 100000, 120000),
'max_iter': 125000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],

'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
'clip': True,
'name': 'VOC_scat',
}


VOC_scat_small = {
'num_classes': 21,
'lr': 3e-4,
'lr_bn': 1e-3,
'lr_steps': (4000, 4500),
'max_iter': 5000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],

'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
'clip': True,
'name': 'VOC_scat_small',
}


kitti300x300_scat = {
'num_classes': 5,
'lr': 1e-4,
'lr_bn': 1e-3,
'lr_steps': (150000, 175000, 185000),
'max_iter': 200000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],

'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [128, 128, 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
'clip': True,
'name': 'kitti_voc_scat',
}


toy_data_scat_parallel = {
'num_classes': 4,
'lr': 1e-4,
'lr_bn': 3e-3,
'lr_steps': (60000, 75000, 85000),
'max_iter': 100000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 4, 4, 4, 4, 4],
'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'toy_data_scat_parallel',
}

toy_data_small_scat_parallel = {
'num_classes': 4,
'lr': 1e-4,
'lr_bn': 3e-3,
'lr_steps': (4000, 4500),
'max_iter': 5000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 4, 4, 4, 4, 4],
'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'toy_data_small_scat_parallel',
}


voc_scat_parallel = {
'num_classes': 21,
'lr': 3e-4,
'lr_bn': 3e-3,
'lr_steps': (80000, 100000, 120000),
'max_iter': 125000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],
'variance': [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'voc_scat_parallel',
}

voc_small_scat_parallel = {
'num_classes': 21,
'lr': 3e-4,
'lr_bn': 3e-3,
'lr_steps': (4000, 4500),
'max_iter': 5000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.33, 0.5, 2, 3],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 6, 6, 6, 4, 4],
'variance': [0.1, 0.2],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'voc_small_scat_parallel',
}


toy_data_small = {
'num_classes': 4,
'lr': 1e-4,
'lr_bn': 1e-3,
#'lr_steps': (80000, 100000, 120000),
#'max_iter': 125000,
'lr_steps':(4000, 4500),
'max_iter':5000,
'feature_maps_y': [38, 19, 10, 5, 3, 1],
'feature_maps_x': [38, 19, 10, 5, 3, 1],
'dim_x': 300,
'dim_y': 300,
'steps': [8, 16, 32, 64, 100, 300],
's_sizes': [30, 60, 111, 162, 213, 264, 315],
'aspect_ratios': [
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2],
[0.5, 2]
],
'mbox': [4, 4, 4, 4, 4, 4],
'variance': [0.1, 0.2],
#    'base': [64, 'M', 128 'M', 256, 'C', 512, 'M',
#            512, 512, 512],
'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
'clip': True,
'name': 'toy_data_small',
}




