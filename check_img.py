# for single image and text file check of yolo format dataset

from PIL import Image,ImageDraw
import numpy as np

def load(anno_path):
    with open(anno_path,'r') as f:
        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
        l = np.array(l, dtype=np.float32)
    return l


def xywhn2xyxy(x, w=512, h=512, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

image_path = '7650.png'
anno_path = '7650.txt'

image = Image.open(image_path)
# draw = ImageDraw.Draw(image)
# annos = load(anno_path)
# annos[:, 1:] = xywhn2xyxy(annos[:, 1:])
# for ann in annos:
#     bbox = ann[1:]
#     x1 = int(bbox[0])
#     y1 = int(bbox[1])
#     x2 = int(bbox[2])
#     y2 = int(bbox[3])
#     label_name = str(ann[0])
#     draw.rectangle([x1, y1, x2, y2], outline='red')
#     draw.text((x1, y1), label_name, (0, 255, 255))
# image.show()
from skimage.util import random_noise