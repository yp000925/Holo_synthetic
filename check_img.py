# for single image and text file check of yolo format dataset

from PIL import Image,ImageDraw,ImageFont
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
#
# image_path = '10.png'
# anno_path = '10.txt'

image_path = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test/5_4.png'
anno_path = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/labels/small_test/5_4.txt'
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
annos = load(anno_path)
annos[:, 1:] = xywhn2xyxy(annos[:, 1:])
font = ImageFont.truetype("/Users/zhangyunping/PycharmProjects/3Ddetection/arial.ttf", 16)
for ann in annos:
    bbox = ann[1:]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    label_name = str(int(ann[0]/255*1000)/1000)
    draw.rectangle([x1, y1, x2, y2], outline='green',width=3)
    draw.text((x1, y1), label_name, 'yellow',font=font)
image.show()
from skimage.util import random_noise

import matplotlib.pyplot as plt

def plot_3d_ouput(labels,ax,color,marker,s=20):
    z = labels[:,0]/255.0 #class,x1,y1,x2,y2
    x = (labels[:,1]+labels[:,3])/2
    y = (labels[:,2]+labels[:,4])/2
    ax.scatter(np.array(x),np.array(y),np.array(z),c=color,marker=marker,s=s)


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.set_zlim(0, 1)
ax.set_xlabel('X (pixels) ',fontsize=16)
ax.set_ylabel('Y (pixels) ',fontsize=16)
ax.set_zlabel('Z (Normolized)',fontsize=16)

plot_3d_ouput(annos,ax,'g','o',s=40)
plot_3d_ouput(annos,ax,'r','^',s=20)

#ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)