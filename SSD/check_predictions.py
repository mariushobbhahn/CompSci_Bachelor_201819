from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import *
from utils.augmentations import SSDAugmentation
import torch.utils.data as data
import cv2
#from imutils.video import FPS, WebcamVideoStream
from ssd import build_ssd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd_VOC_300x300_random_no_batch_norm_pretrained_13_125000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--config', default='300x300', choices=['300x300', '1000x300'],
                    type=str, help='size of the imagescales')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--normalize', default=False, type=str2bool,
                    help='normalize images before training')
parser.add_argument('--subtract_mean', default=True, type=str2bool,
                    help='subtract the color means before training')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'kitti_voc', 'kitti_voc_small', 'toy_data'],
                    type=str, help='VOC, kitti_voc, toy_data or kitti_voc_small')


args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.dataset == 'VOC':
    annopath = os.path.join(args.dataset_root, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(args.dataset_root, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(args.dataset_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
    YEAR = '2007'
    devkit_path = args.dataset_root + 'VOC' + YEAR
    dataset_mean = VOC_MEANS
    set_type = 'test'
    labelmap = VOC_CLASSES

elif args.dataset == 'kitti_voc':
    annopath = os.path.join(args.dataset_root,  'Annotations', '%s.xml')
    imgpath = os.path.join(args.dataset_root,  'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(args.dataset_root, 'ImageSets',
                          'Main', '{:s}.txt')
    devkit_path = args.dataset_root
    dataset_mean = KITTI_MEANS
    set_type = 'val'
    labelmap = kitti_CLASSES

elif args.dataset == 'kitti_voc_small':
    annopath = os.path.join(args.dataset_root,  'Annotations', '%s.xml')
    imgpath = os.path.join(args.dataset_root,  'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(args.dataset_root, 'ImageSets',
                          'Main', '{:s}.txt')
    devkit_path = args.dataset_root
    dataset_mean = KITTI_MEANS
    set_type = 'train'
    labelmap = kitti_CLASSES

elif args.dataset == 'toy_data':
    annopath = os.path.join(args.dataset_root,  'Annotations', '%s.xml')
    imgpath = os.path.join(args.dataset_root,  'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(args.dataset_root, 'ImageSets',
                          'Main', '{:s}.txt')
    devkit_path = args.dataset_root
    dataset_mean = (0,0,0)
    set_type = 'test'
    labelmap = toy_data_CLASSES


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))



COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def show_label_annotations(image_file, label_file, save_file=None):
    #get the image
    image = cv2.imread(image_file)

    #get the annotations
    tree = parse_rec(label_file)

    for i in range(len(tree)):
        #add a box for every item in tree
        item = tree[i]
        pts = item['bbox']
        name = item['name']
        cv2.rectangle(image,
                      (int(pts[0]), int(pts[1])),
                      (int(pts[2]), int(pts[3])),
                      COLORS[i % 3], 2)
        cv2.putText(image, name, (int(pts[0]), int(pts[1])), FONT, 2, COLORS[i % 3], 2, cv2.LINE_AA) # (255, 255, 255)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    if save_file != None:
        print('saving image')
        #cv2.imwrite('Images/annotations_for_{}.png'.format(image_file), image)
        cv2.imwrite('Images/annots.png', image)
    cv2.destroyAllWindows()
    #print(tree)



def predict(net, dataset, frame_index, original_size=False, original_image=False, confidence_threshold=0.5):

    im, gt, height, width = dataset.pull_item(frame_index)

    x = Variable(im.unsqueeze(0))
    if args.cuda:
        x = x.cuda()

    _t = {'im_detect': Timer(), 'misc': Timer()}
    _t['im_detect'].tic()
    detections = net(x).data
    print("detections_size: ", detections.size())
    #print("detections: ", detections)
    detect_time = _t['im_detect'].toc(average=False)

    im = im.permute(1, 2, 0).cpu()
    im = im.numpy().copy()

    if not original_size:
        print("image size: ", np.shape(im))
        width = np.shape(im)[1]
        height = np.shape(im)[0]
    else:
        im = cv2.resize(im, (width, height))

    scale = torch.Tensor([width, height, width, height])
    print("scale: ", scale)
    # skip j = 0, because it's the background class
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= confidence_threshold:
            print("confidence:", detections[0, i, j, 0])
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            print("pt: ", pt)
            cv2.rectangle(im,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(im, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 2, COLORS[i % 3], 2, cv2.LINE_AA)
            j += 1


            #print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

    return(im)


if __name__ == '__main__':
    # load net
    if args.dataset == 'VOC':
        cfg = voc

        dataset = VOCDetection(root = VOC_ROOT,
                               image_sets=[('2007', set_type)],
                               transform = BaseTransform(size_x=cfg['dim_x'],
                                                         size_y=cfg['dim_y'],
                                                         sub_mean=args.subtract_mean,
                                                         mean=VOC_MEANS,
                                                         normalize=False,
                                                         norm_value=255),
                               target_transform=VOCAnnotationTransform())

    elif args.dataset == 'toy_data':
        cfg = voc  #toy_data

        dataset = toydataDetection(root = toy_data_ROOT,
                              image_sets=[set_type],
                              transform = BaseTransform(size_x=cfg['dim_x'],
                                                        size_y=cfg['dim_y'],
                                                        sub_mean=args.subtract_mean,
                                                        mean=(0,0,0),
                                                        normalize=args.normalize,
                                                        norm_value=255),
                              target_transform=toydataVOCAnnotationTransform())

    elif args.dataset == 'kitti_voc':
        if args.config == '300x300':
            cfg = kitti300x300
        elif args.config == '1000x300':
            cfg = kitti1000x300
        else:
            raise ValueError('The given configuration is not possible')

        dataset = kittiDetection(root = kitti_ROOT,
                                image_sets=[set_type],
                                transform = BaseTransform(size_x=cfg['dim_x'],
                                                          size_y=cfg['dim_y'],
                                                          sub_mean=args.subtract_mean,
                                                          mean=KITTI_MEANS,
                                                          normalize=args.normalize,
                                                          norm_value=255),
                                target_transform=kittiVOCAnnotationTransform())

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
                                                            eval=True))

    else:
        raise(TypeError("config was not possible. check for typos"))


    #test_image = "/home/marius/data/kitti_voc/JPEGImages/000000001_000468.jpg"
    #test_label = "/home/marius/data/kitti_voc/Annotations/000000001_000468.xml"
    #test_image = "/home/marius/data/kitti_voc/JPEGImages/000000032_007469.jpg"
    #test_label = "/home/marius/data/kitti_voc/Annotations/000000032_007469.xml"
    #test_image_toy_data = "/home/marius/data/toy_data/JPEGImages/0010000.jpg"
    #test_label_toy_data = "/home/marius/data/toy_data/Annotations/0010000.xml"
    #show_label_annotations(test_image_toy_data, test_label_toy_data, save_file=False)
    #cv2.imwrite('Images/test_annotation.png',pred)

    print("dim_x, y: ", cfg['dim_x'], cfg['dim_y'])
    net = build_ssd(phase='test', size_x=cfg['dim_x'], size_y=cfg['dim_y'], num_classes=cfg['num_classes'], cfg=cfg, batch_norm=False)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True


    pred = predict(net=net.eval(), dataset=dataset, frame_index=6, original_size=True, confidence_threshold=0.30)
    cv2.imshow('frame', pred)
    #cv2.imshow("image", imagefile)
    cv2.waitKey(0)
    #cv2.imwrite('Images/test_VOC.png',pred)
    cv2.destroyAllWindows()

    #"""
