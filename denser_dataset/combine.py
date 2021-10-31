# this script for combine the dataset into a whole

import json
import pandas as pd
import numpy

import os
# path = "/Users/zhangyunping/PycharmProjects/Holo_synthetic/denser_dataset/512_250_particle_train.json"
images = []
annotations = []
categories = []
info = []
licenses = []


files = os.listdir("/Users/zhangyunping/PycharmProjects/Holo_synthetic/denser_dataset")

for file in files:
    if file.endswith("json"):
        with open("/Users/zhangyunping/PycharmProjects/Holo_synthetic/denser_dataset/"+file, 'r') as f:
            data = json.load(f)
            images.extend(data['images'])
            categories.extend(data['categories'])
            annotations.extend(data['annotations'])
licenses.extend(data['licenses'])
info.extend(data['info'])


dataset = {}
dataset['images'] = images
dataset['info'] = info
dataset['licenses'] = licenses
dataset['categories'] = categories
dataset['annotations'] = annotations

json_name = "/Users/zhangyunping/PycharmProjects/Holo_synthetic/denser_dataset/denser_data.json"
with open(json_name,'w') as f:
    json.dump(dataset,f)
#%% particle split
import os
import glob
pngfile = []
txtfile = []
for f_name in glob.glob('/home/ypzhang/exp/holo_data/0920_250_particle/img/*.png'):
    pngfile.append(f_name)
for f_name in glob.glob('/home/ypzhang/exp/holo_data/0920_250_particle/param/*.txt'):
    txtfile.append(f_name)
