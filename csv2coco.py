from utils import *
import numpy as np
import  pandas as pd
import  time

import json
import os
import re

wavelength = 633 * nm
N = 1024
# pixel_pitch = 10*um
frame = 10 * mm  # 10mm * 10mm
# size_range = [20 * um, 100 * um]
size_range = [50*um,50*um]
res_z =  (3*cm-1*cm)/256

def get_buffer(z,size):
    z_rate = (z-1*cm)/(3*cm-1*cm)
    # size_rate = (size-size_range[0])/(size_range[1]-size_range[0])
    size_rate = 1
    buffer = 30+30*z_rate+15*size_rate
    # p_size = int(size_range[1]/frame * N)
    # buffer = p_size*10*(z_rate*0.6+size_rate*0.4)
    return buffer

def get_bbox(x,y,z,size):
    px = int(x/frame*N+N/2)
    py = int(N/2+y/frame*N)
    # p_size = int(size/frame*N)
    # buffer = p_size*10
    buffer = get_buffer(z,size)
    bbox_x = max(0, px-buffer)
    bbox_y = max(0, py-buffer)
    height = buffer*2
    width = buffer*2
    if bbox_x+width > N:
        width = N-bbox_x
    if bbox_y+height > N:
        height = N-bbox_y
    seg = [bbox_x,bbox_y,bbox_x,bbox_y+height,bbox_x+width,bbox_y+height,bbox_x+width,bbox_y]
    return (bbox_x,bbox_y,width,height,seg)

def collect_anno(filename):
    id = re.findall(r'\d+', filename)
    id = int(id[0])
    df = pd.read_csv('/Users/zhangyunping/PycharmProjects/Holo_synthetic/test_data/param/'+filename)
    return (id,df)


dataset={}
dataset['info'] = []
dataset['licenses'] = []
dataset['info'].append({'year':"2021", "version": '1',
        "description": "Hologram synthetic data for test 100ppp",
        "contributor": "zhangyp",
        "url": "None",
        "date_created": "2021-08-05"})
dataset['licenses'].append({
            "id": 1,
            "url": "None",
            "name": "zhangyp"
        })

dataset['categories']=[]
dataset['images']=[]
dataset['annotations']=[]

# build the category based on the depth
classes = list(range(1,257))
# classes = np.array(np.linspace(1*cm, 3*cm, 256))


for cls in classes:
    dataset['categories'].append({'id': int(cls), 'name': str(cls), 'supercategory':'Depth'})


images = np.sort(os.listdir('/Users/zhangyunping/PycharmProjects/Holo_synthetic/test_data/hologram'))



for i, imagename in enumerate(images, 0):
    id = re.findall(r'\d+', imagename)
    dataset['images'].append(({'id':int(id[0]), 'width':1024, 'height':1024, 'file_name':imagename, 'license':'None'}))
    # if i == 10:
    #     break

params = np.sort(os.listdir('/Users/zhangyunping/PycharmProjects/Holo_synthetic/test_data/param'))
cnt_annot = 20000000
for i,param in enumerate(params,0):
    if param == '.DS_Store':
        continue
    (id,objs) = collect_anno(param)
    for index, obj in objs.iterrows():
        (bbox_x, bbox_y, width, height, seg) = get_bbox(obj['x'], obj['y'], obj['z'], obj['size'])
        depth = obj['z']
        category_id = int((depth-1*cm)/res_z)+1

        if category_id == 256:
            category_id = 255

        # if category_id == 0:
        #     category_id = 1

        dataset['annotations'].append({
            'area': width*height,
            'bbox' : [bbox_x, bbox_y, width, height],
            'category_id': category_id,
            'id': cnt_annot,
            'image_id': int(id),
            'iscrowd': 0,
        })
        cnt_annot += 1
    # if i == 0:
    #     break

json_name = 'annotations_fortest_100.json'
with open(json_name,'w') as f:
    json.dump(dataset,f)


