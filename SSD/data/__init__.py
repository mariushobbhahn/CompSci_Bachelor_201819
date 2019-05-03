from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .kitti import kittiDetection, kittiVOCAnnotationTransform, kitti_CLASSES, kitti_ROOT
from .kitti_small import kitti_small_Detection, kitti_small_ROOT
from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .toy_data import toydataDetection, toydataVOCAnnotationTransform, toy_data_CLASSES, toy_data_ROOT
from .rotation_data import rotationdataDetection, rotationdataVOCAnnotationTransform, rotation_data_CLASSES, rotation_data_ROOT
from .scale_data import scaledataDetection, scaledataVOCAnnotationTransform, scale_data_CLASSES, scale_data_ROOT
from .deformation_data import deformationdataDetection, deformationdataVOCAnnotationTransform, deformation_data_CLASSES, deformation_data_ROOT
from .translation_data import translationdataDetection, translationdataVOCAnnotationTransform, translation_data_CLASSES, translation_data_ROOT
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size_x, size_y, sub_mean, mean, normalize, norm_value):
    x = cv2.resize(image, (size_x, size_y)).astype(np.float32)
    if sub_mean:
        x -= mean
    if normalize:
        x = x/norm_value
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size_x, size_y, mean, sub_mean=True, normalize=True, norm_value=255):
        self.size_x = size_x
        self.size_y = size_y
        self.sub_mean = sub_mean
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.norm_value = np.array(norm_value, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size_x, self.size_y, self.sub_mean, self.mean, self.normalize, self.norm_value), boxes, labels
