from PIL import Image
import numpy as np

import sys
import os
import torch
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
import pandas as pd
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
from utils import *
import re

def get_new_bbox(old_bbox, boundary):
    # old bbox 里面是以左上为原点
    [o_x1, o_y1, o_x2, o_y2] = old_bbox[0], old_bbox[1],old_bbox[0]+old_bbox[2],old_bbox[1]+old_bbox[3]
    [b_x1, b_y1, b_x2, b_y2] = boundary[0], boundary[2],boundary[1],boundary[3]
    o_cx,o_cy = (o_x2+o_x1)/2,(o_y1+o_y2)/2

    if b_x1<o_cx<b_x2 and b_y1<o_cy<b_y2:
        n_x1 = max(o_x1, b_x1)
        n_y1 = max(o_y1, b_y1)
        n_x2 = min(o_x2, b_x2)
        n_y2 = min(o_y2, b_y2)

        n_w = n_x2-n_x1
        n_h = n_y2-n_y1

        if n_w <=0 or n_h<=0:
            return []
        else:
            # axis center transfer
            n_x1 = n_x1-b_x1
            n_y1 = n_y1-b_y1
            return [n_x1,n_y1,n_w,n_h]
    else:
        return []

def get_new_annos(old_annos,boundary,img_id):
    global ANNO_CNT
    new_annos = []
    for anno in old_annos:
        new_bbox = get_new_bbox(anno['bbox'],boundary)
        if len(new_bbox) != 4:
            continue
        new_annos.append({
            'area': new_bbox[2]*new_bbox[3],
            'bbox': new_bbox,
            'category_id': anno['category_id']+1,
            'id': ANNO_CNT,
            'image_id': img_id,
            'iscrowd': 0,
        })
        ANNO_CNT += 1
    return new_annos

if __name__ == "__main__":
    import time

    ANNO_CNT = 1
    dataset = {}
    dataset['info'] = []
    dataset['licenses'] = []
    dataset['holoinfo'] = []
    dataset['info'].append({'year': "2021", "version": '0',
                            "description": "Hologram synthetic data clipped version",
                            "contributor": "zhangyp",
                            "url": "None",
                            "date_created": "2021-05-29",
                            })
    dataset['licenses'].append({
        "id": 1,
        "url": "None",
        "name": "zhangyp"
    })

    dataset['categories'] = []
    dataset['images'] = []
    dataset['annotations'] = []
    classes = list(range(1, 257))
    # classes = np.array(np.linspace(1*cm, 3*cm, 256))
    for cls in classes:
        dataset['categories'].append({'id': int(cls), 'name': str(cls), 'supercategory': 'Depth'})

    wavelength = 633 * nm
    N = 1024
    # pixel_pitch = 10*um
    size = 10 * mm  # 10mm * 10mm
    size_range = [20 * um, 100 * um]
    res = size / N
    anno_path = '_annotations_holoall.json'
    coco = COCO(annotation_file=anno_path)
    categories = coco.loadCats(coco.getCatIds())
    img_ids = coco.getImgIds()
    print("total image number in the dataset is {:d}".format(len(img_ids)))
    root_dir = '/Users/zhangyunping/PycharmProjects/3Ddetection/data/hologram'
    IMG_ID = 0 # each image is linked to an unique id
    crop_w, crop_h = 512, 512
    stride = 256

    for i in range(len(img_ids)):
        # Read image
        image_info = coco.loadImgs(img_ids[i])[0]
        path = os.path.join(root_dir, image_info['file_name'])
        # img = np.array(Image.open(path))
        # Read annotations
        annotation_ids = coco.getAnnIds(imgIds=img_ids[i], iscrowd=False)
        old_annos = coco.loadAnns(annotation_ids)

        # index for sub image
        idx = 0
        for c_step in range(int((N - crop_h) // stride + 1)):
            for r_step in range(int((N - crop_w) // stride + 1)):
                boundary = [int(stride * r_step), int(stride * r_step + crop_w), int(stride * c_step),
                        int(stride * c_step + crop_h)]
                # boundary = [x0,x1,y0,y1]
                if boundary[3] <= N and boundary[1] <= N:
                    # print(bbox)
                    # clipped_img = img[boundary[2]:boundary[3], boundary[0]:boundary[1], :]
                    d = 1

                else:
                    continue
                # save the cropped image
                name = str(img_ids[i]) + '_' + str(idx) + '.png'
                # clipped_img = Image.fromarray((clipped_img / np.max(clipped_img) * 255).astype(np.uint8))
                # clipped_img.save('clipped_data/holo' + '/' + name)
                dataset['images'].append(
                    ({'id': IMG_ID, 'width': crop_w, 'height': crop_h, 'file_name': name, 'license': 'None'}))
                # get the clipped bbox label
                new_annos = get_new_annos(old_annos, boundary, IMG_ID)
                dataset['annotations'].extend(new_annos)
                IMG_ID += 1
                idx += 1
        if i % 500 == 0:
            print(i, IMG_ID, time.ctime())


    import json
    json_name = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/annotations_clip_512_2nd.json'
    with open(json_name,'w') as f:
        json.dump(dataset,f)
#
    from PIL import Image,ImageDraw

    # Draw RLE label
    coco = COCO(annotation_file="/clipped_data/annotations_clip_512_2nd.json")
    img_ids = coco.getImgIds()
    annotation_ids = coco.getAnnIds(imgIds = img_ids[5])
    annos = coco.loadAnns(annotation_ids)
    image_info = coco.loadImgs(img_ids[5])
    image_path = image_info[0]["file_name"]
    image_path = os.path.join("/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/train", image_path)
    print("image path (crowd label)", image_path)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for ann in annos:
        bbox = ann['bbox']
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[0]+bbox[2])
        y2 = int(bbox[1]+bbox[3])
        label_name = str(ann['category_id'])
        draw.rectangle([x1, y1, x2, y2], outline='red')
        draw.text((x1, y1), label_name, (0, 255, 255))
