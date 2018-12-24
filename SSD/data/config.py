# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

VOC_MEANS = (102.93, 111.36, 116.52)
KITTI_MEANS = (96.19429688, 95.55136972, 91.34692262)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (50000, 75000, 100000),
    'max_iter': 125100,
    'feature_maps_y': [38, 19, 10, 5, 3, 1],
    'feature_maps_x': [38, 19, 10, 5, 3, 1],
    'dim_x': 300,
    'dim_y': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    's_sizes': [30, 60, 111, 162, 213, 264, 315],
    #'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_ratios': [
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3]
    ],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# coco = {
#     'num_classes': 201,
#     'lr_steps': (280000, 360000, 400000),
#     'max_iter': 400000,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'dim_x': 300,
#     'dim_y': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [21, 45, 99, 153, 207, 261],
#     'max_sizes': [45, 99, 153, 207, 261, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'COCO',
# }

kitti300x300 = {
    'num_classes': 5,
    'lr_steps': (25000, 50000, 75000),
    'max_iter': 100100,
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
    'aspect_ratios' : [
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
    ],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name': 'kitti300x300',
}

kitti1000x300 = {
    'num_classes': 5,                                   # number of Classes
    'lr_steps': (25000, 50000, 75000),                  # at which episode is the learning rate reduce by factor gamma
    'max_iter': 100100,                                 # amounts of iterations
    'feature_maps_y': [38, 19, 10, 5, 3, 1],            # number of feature maps in y direction
    'feature_maps_x': [125, 62, 31, 16, 14, 12],        # number of feature maps in x direction
    #'dim_x': 300,
    'dim_x': 1000,                                      # x size of image after rescaling
    'dim_y': 300,                                       # y size of image after rescaling
    'steps': [8, 16, 32, 64, 100, 300],
    #'s_sizes_y': [10, 20, 45, 80, 130, 210, 270],       #for self-chosen values
    's_sizes': [30, 60, 105, 150, 195, 240, 285],      #for s_min = 0.2 and s_max = 0.95
    #'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_ratios' : [
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
        [0.33, 0.5, 2, 3],
    ],
    'variance' : [0.1, 0.2],
    'clip' : True,
    'name': 'kitti1000x300',
}
