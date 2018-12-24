import argparse
import sys
import cv2
import os
from data import VOC_CLASSES, kitti_CLASSES
import os.path          as osp
import numpy            as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree  as ET


parser    = argparse.ArgumentParser(
            description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

#parser.add_argument('--root', help='Dataset root directory path', default='/home/marius/data/kitti_voc')
parser.add_argument('--root', help='Dataset root directory path', default='/home/marius/data/VOCdevkit/VOC2007')
parser.add_argument('--dataset', help='Dataset is either kitti_voc or VOC', default='kitti_voc')

args = parser.parse_args()

if args.dataset == 'kitti_voc':
    CLASSES = kitti_CLASSES
elif args.dataset == 'VOC':
    CLASSES = VOC_CLASSES

annopath = osp.join('%s', 'Annotations', '%s.{}'.format("xml"))
imgpath  = osp.join('%s', 'JPEGImages',  '%s.{}'.format("jpg"))

def vocChecker(image_id, width, height, keep_difficult = False):
    target   = ET.parse(annopath % image_id).getroot()
    res      = []

    for obj in target.iter('object'):

        difficult = int(obj.find('difficult').text) == 1

        if not keep_difficult and difficult:
            continue

        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts    = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []

        for i, pt in enumerate(pts):

            cur_pt = int(bbox.find(pt).text) - 1
            #print("cur_pt: ", cur_pt)
            # scale height or width
            #cur_pt = float(cur_pt) / width if i % 2 == 0 else float(cur_pt) / height

            bndbox.append(cur_pt)

        #print(name)
        #label_idx =  dict(zip(CLASSES, range(len(CLASSES))))[name]
        #bndbox.append(label_idx)
        bndbox = [bndbox[2] - bndbox[0], bndbox[3] - bndbox[1]]  #, bndbox[4]]         # change absolute values to relative ones through xmax-xmin, ymax- ymin
        x_300, y_300 = int(bndbox[0] * (300/height)), int(bndbox[1] * (300/width))
        #print("unscaled, scaled x: ", bndbox[0], x_300)
        bndbox = [x_300, y_300]            # multiply values such that it fits on a 300x300 image mapping
        #print("bbox: ", bndbox)
        res.append(bndbox)  # [xmin, ymin, xmax, ymax, label_ind]


    return(res)


if __name__ == '__main__' :

    i = 0
    list_of_tuples = []

    for name in sorted(os.listdir(osp.join(args.root,'Annotations'))):
    # as we have only one annotations file per image
        i += 1

        img    = cv2.imread(imgpath  % (args.root,name.split('.')[0]))
        height, width, channels = img.shape
        #print("height, width: ", height, width)

        res = vocChecker((args.root, name.split('.')[0]), height, width)
        list_of_tuples.extend(res)

        #print("x-size, y-size: ", res)

        #print("path : {}".format(annopath % (args.root,name.split('.')[0])))
    print("Total number of annotations : {}".format(i))
    print("number of ratios: ", len(list_of_tuples))
    #unique_list_of_tuples = np.unique(list_of_tuples, axis=0)
    #print("number of unique tuples: ", len(list_of_tuples))
    #print("unique list of tuples: ", list_of_tuples)
    x_entries = np.array(list_of_tuples)[:, 0]
    y_entries = np.array(list_of_tuples)[:, 1]
    ratios = x_entries/y_entries
    print("unique list of x_entries: ", x_entries)
    print("unique list of y_entries: ", y_entries)
    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True)
    axs[0].hist(x_entries)
    axs[1].hist(y_entries)
    axs[2].hist(ratios)
    plt.show()
