import os
import numpy as np
import re

img_s_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images'

label_s_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/labels/annotations_clip_512'

data_list = os.listdir(img_s_dir)
data_list.sort()
data_list = [x  for x in data_list if re.match('[\w]*\.png$',x)]

train_data = data_list[0:int(len(data_list)*0.7)]
validation_data = data_list[int(len(data_list)*0.7):int(len(data_list)*0.7)+int(len(data_list)*0.2)]
test_data = data_list[int(len(data_list)*0.7)+int(len(data_list)*0.2):]

img_d_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/train'
label_d_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/labels/train'

for dir in [img_d_dir,label_d_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)

for img_name in train_data:
    try:
        old_img_name = img_s_dir+'/'+img_name
        new_img_name = img_d_dir+'/'+img_name
        os.rename(old_img_name,new_img_name)

        img_ind = img_name.split(".")[0]
        old_label_name = label_s_dir+'/'+img_ind + '.txt'
        new_label_name = label_d_dir+'/'+img_ind + '.txt'
        os.rename(old_label_name, new_label_name)
    except:
        print(img_name)
        continue

img_d_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/validation'
label_d_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/labels/validation'
for dir in [img_d_dir,label_d_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)

for img_name in validation_data:
    try:
        old_img_name = img_s_dir+'/'+img_name
        new_img_name = img_d_dir+'/'+img_name
        os.rename(old_img_name,new_img_name)

        img_ind = img_name.split(".")[0]
        old_label_name = label_s_dir+'/'+img_ind + '.txt'
        new_label_name = label_d_dir+'/'+img_ind + '.txt'
        os.rename(old_label_name, new_label_name)
    except:
        print(img_name)
        continue



img_d_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/test'
label_d_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/labels/test'
for dir in [img_d_dir,label_d_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)

for img_name in test_data:
    try:
        old_img_name = img_s_dir+'/'+img_name
        new_img_name = img_d_dir+'/'+img_name
        os.rename(old_img_name,new_img_name)

        img_ind = img_name.split(".")[0]
        old_label_name = label_s_dir+'/'+img_ind + '.txt'
        new_label_name = label_d_dir+'/'+img_ind + '.txt'
        os.rename(old_label_name, new_label_name)
    except:
        print(img_name)
        continue

