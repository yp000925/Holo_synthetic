'''
generate the random wolk for the 3D field without overlapping
'''

from utils import *
from propagators import ASM
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
from tqdm import tqdm
import numpy as np
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import trackpy as tp
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

dims = 3

step_n = 30 # the number of step
step_set = [10, 10, 20] #for x y z direction

def get_origin(xyrange=None, zrange=None, dims =3):
    if xyrange is None:
        xyrange = [-256, 256]
    if zrange is None:
        zrange = [-64, 64]

    if dims == 3:
        xy_origin = np.random.randint(low=xyrange[0], high=xyrange[1], size=(2))
        z_origin = np.random.randint(low=zrange[0],high=zrange[1], size=(1))
        origin = np.concatenate((xy_origin, z_origin))
    else:
        origin=np.random.randint(low=xyrange[0], high=xyrange[1], size=(2))
    return origin

def generate_one_track(step,xyrange=None, zrange=None, dims =3):
    if xyrange is None:
        xyrange = [-256, 256]
    if zrange is None:
        zrange = [-64, 64]
    origin = get_origin(xyrange=xyrange,zrange=zrange)

    current_pos = origin

    hitx = False
    hity = False
    hitz = False
    path = []
    for s in range(step):
        path.append(current_pos)
        step_prob = np.random.uniform(0, 1, dims)
        if hitx:
            next_pos = np.array([current_pos[0] - step_set[0] if current_pos[0] >= xyrange[1] else current_pos[0] + step_set[0], current_pos[1],
                                 current_pos[2]])
            hitx = False
        else:
            if step_prob[0]<=0.3:
                next_pos = np.array([current_pos[0]-step_set[0],current_pos[1],current_pos[2]])
            elif 0.3<step_prob[0]<0.6:
                next_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
            else:
                next_pos = np.array([current_pos[0]+step_set[0], current_pos[1], current_pos[2]])

        if hity:
            next_pos = np.array([next_pos[0], current_pos[1] - step_set[1] if current_pos[1] >= xyrange[1] else current_pos[1] + step_set[1],
                                 current_pos[2]])
            hity=False
        else:
            if step_prob[1]<=0.3:
                next_pos = np.array([next_pos[0], current_pos[1]-step_set[1], current_pos[2]])
            elif 0.3<step_prob[1]<0.6:
                next_pos = np.array([next_pos[0], current_pos[1], current_pos[2]])
            else:
                next_pos = np.array([next_pos[0], current_pos[1]+step_set[1], current_pos[2]])

        if hitz:
            next_pos = np.array([next_pos[0], next_pos[1],
                                 current_pos[2] - step_set[2] if current_pos[2] >= zrange[1] else current_pos[2] + step_set[2]])
            hitz=False
        else:
            if step_prob[1]<=0.3:
                next_pos = np.array([next_pos[0],next_pos[1], current_pos[2]-step_set[2]])
            elif 0.3<step_prob[1]<0.6:
                next_pos = np.array([next_pos[0],next_pos[1], current_pos[2]])
            else:
                next_pos = np.array([next_pos[0],next_pos[1], current_pos[2]+step_set[2]])


        if next_pos[0] <= xyrange[0] or next_pos[0] >= xyrange[1]:
            hitx= True
            current_pos[0] = xyrange[0] if next_pos[0] <= xyrange[0] else xyrange[1]

        if next_pos[1] <= xyrange[0] or next_pos[1] >= xyrange[1]:
            hity= True
            current_pos[1] = xyrange[0] if next_pos[1] <= xyrange[0] else xyrange[1]

        if next_pos[2] <= zrange[0] or next_pos[2] >= zrange[1]:
            hitz = True
            current_pos[2] = zrange[0] if next_pos[2] <= zrange[0] else zrange[1]

        if not hitx and not hity and not hitz:
            current_pos = next_pos
    return np.array(path)


# ini_tracks = pd.DataFrame(columns = ['x','y','z','frame','particle_idx'])
# for n in tqdm(range(0,n_runs)):
#     path = generate_one_track(step_n,xyrange=[-40,40])
#     buffer = pd.DataFrame(columns= ['x','y','z','frame','particle_idx'])
#     buffer['x'] = path[:,0]
#     buffer['y'] = path[:,1]
#     buffer['z'] = path[:,2]
#     buffer['frame'] = np.array(range(len(path)))
#     buffer['particle_idx'] = [n]*len(path)
#     ini_tracks = pd.concat([ini_tracks,buffer],axis=0)


box_w = 70
box_h = 70
stride = 70
N_x = 512
N_y = 512
particle_idx = 0
tracks = pd.DataFrame(columns = ['x','y','z','frame','particle_idx'])
for c_step in tqdm(range((N_x-box_w)//stride+1)):
    for r_step in range((N_y-box_h)//stride+1):
        flag = np.random.rand(1) < 0.7
        if flag:
            continue
        buffer = pd.DataFrame(columns= ['x','y','z','frame','particle_idx'])
        path = generate_one_track(step_n, xyrange=[-50, 50])
        c_x = -512/2 + box_w/2 + c_step*stride
        c_y = -512/2 + box_h/2 + r_step*stride
        # idx = np.random.choice(ini_tracks.particle_idx.unique(),1)
        # particle_info = ini_tracks[ini_tracks["particle_idx"]==idx[0]]
        buffer['x'] = path[:,0] + c_x
        buffer['y'] = path[:,1] + c_y
        buffer['z'] = path[:,2]
        buffer['particle_idx']  = [particle_idx]*len(path)
        buffer['frame'] = np.array(range(len(path)))
        tracks = pd.concat([tracks,buffer],axis=0)
        particle_idx += 1

# tracks.to_csv("3d_tracks_no_overlap.csv",index = False)
# gt = tracks
# fig = plt.figure()
# ax = fig.add_subplot(111,projection = '3d')
# particle_idx = gt['particle_idx'].unique()

# from itertools import cycle
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for i,(idx,col) in enumerate(zip(particle_idx,colors)):
#      path = gt[gt['particle_idx'] == idx]
#      ax.plot3D(path['x'],path['y'],path['z'],c=col)
#      if i == 10:
#           break
# plt.show()