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
from PIL import Image

class CropdataUtil():
    """
    Dataset for 3D particle detection using capsule net
    """
    def __init__(self, root_dir, file_name = 'train_data.csv', transform = None, size=1024):
        '''
        :param holo_dir: directory for holograms
        :param depthmap_dir: directory for depthmap
        :param xycentre_dir: directory for xycentre
        :param file_name: file_name
        :param transform:
        '''
        # self.holo_dir = 'holo_dir'
        # self.depthmap_dir = 'depthmap_dir'
        # self.xycentre_dir = xycentre_dir
        self.root_dir = root_dir
        self.file_name = file_name
        self.transform = transform
        self.file = pd.read_csv(os.path.join(root_dir,file_name))
        self.N =size

    def __getitem__(self, idx):
        data = self.file.iloc[idx]
        holo_path = os.path.join(self.root_dir, 'hologram', data['hologram'])
        param_path = os.path.join(self.root_dir, 'param', data['param'])
        img = self.read_img(holo_path)
        param = self.load_param(param_path)
        size_projection,xycentre,xy_mask = self.get_maps(param)
        return img,size_projection,xycentre,xy_mask

    def get_maps(self,param):
        size_projection, xy_mask = self.get_xy_projection(param)
        xycentre = self.get_xycentre(param)
        return (size_projection,xycentre,xy_mask)

    def get_xy_projection(self,param):
        """
        :param param: px,py,pz,psize stored in dataframe
        :return: map: the xy_projection map, the pixel value is the corresponding depth, range from 0-1
                 mask: the indication map for overlapping 0: the overlap exists -> ignored when calculate the loss
        """
        arr = np.zeros((256,self.N,self.N))
        particle_field = np.zeros(arr.shape) # one stands for the exist of particle

        for _,particle in param.iterrows():
            px,py,pz,psize = particle.x,particle.y,particle.z,particle.size
            Y, X = np.mgrid[:self.N, :self.N]
            Y = Y - py
            X = X - px
            dist_sq = Y ** 2 + X ** 2
            z_slice = np.zeros((self.N,self.N))
            particle_field_slice = np.zeros((self.N,self.N))
            z_slice[dist_sq <= psize ** 2] = pz
            particle_field_slice[dist_sq <= psize ** 2] = 1
            arr[pz,:,:] += z_slice # 可能某个depth上面有多个particles
            particle_field[pz,:,:] += particle_field_slice

        map = arr.sum(axis=0)/255.0
        # check whether there are overlapping
        particle_field_proj = particle_field.sum(axis=0)
        mask_map = np.ones((self.N,self.N))
        mask_map[particle_field_proj>1] = 0 #在后面计算loss的时候，只计算没有overlap的pixel，即mask里面为0的情况忽略
        return map, mask_map

    def get_xycentre(self,param):
        arr = np.zeros((self.N, self.N))
        idx_x = np.array(param['x'].values)
        idx_y = np.array(param['y'].values)
        arr[(idx_y,idx_x)] = 1.0
        return arr

    def load_param(self,param_path):
        param = pd.read_csv(param_path)
        x = param['x'].values
        y = param['y'].values
        z = param['z'].values
        size = param['size']
        frame = 10 * 1e-3
        N=1024
        xyres = frame/N
        px = (x / frame * N + N / 2).astype(np.int)
        py = (N / 2 + y / frame * N).astype(np.int)
        pz = ((z - 1 * 1e-2)/ (3 * 1e-2 - 1 * 1e-2)*255).astype(np.int)
        psize = (size/xyres).astype(np.int)
        param_pixel = pd.DataFrame()
        param_pixel['x'] = px
        param_pixel['y'] = py
        param_pixel['z'] = pz
        param_pixel['size'] = psize
        return param_pixel


    def read_img(self,img_name):
        img = Image.open(img_name)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img).astype(np.float32)
        return img/255.

def random_crop_inputs_and_labels(crop_w, crop_h, stride, *maps):
    size = maps[0].shape
    h = size[0]
    w = size[1]
    grouped_maps = []
    for map in maps:
        cropped_maps = []

        for c_step in range(int((h-crop_h)//stride+1)):
            for r_step in range(int((w-crop_w)//stride+1)):
                bbox = [int(stride*r_step), int(stride*(r_step+1)+crop_w-1), int(stride*c_step), int(stride*(c_step+1)+crop_h-1)] # bbox = [x0,x1,y0,y1]
                print(bbox)
                cropped_map = map[bbox[2]:bbox[3], bbox[0]:bbox[1]]
                cropped_maps.append(cropped_map)

        cropped_maps = np.array(cropped_maps)
        grouped_maps.append(cropped_maps)
    return grouped_maps

    



if __name__ == "__main__":
    # root_dir ='/Users/zhangyunping/PycharmProjects/Holo_synthetic/data_holo'
    # file_path = 'check.csv'
    # dataset = CropdataUtil(root_dir,file_path)
    # for data in dataset:
    #     break

    # test the crop functions
    test = list(range(25))
    test = np.array(test).reshape((5,5))
    croped = random_crop_inputs_and_labels(2,2,1,test)
    for i in range(croped[0].shape[0]):
        print(croped[0][i])

