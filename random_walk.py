
from utils import *
from propagators import ASM
import numpy as np
import pandas as pd
import time
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
import os
from tqdm import tqdm

from itertools import cycle


# Define parameters for the walk
dims = 3
n_runs = 5 # the number of particles
step_n = 200 # the number of step
step_set = [10,10,5] #for x y z direction

def get_origin(xyrange=None, zrange=None, dims =3):
    if xyrange is None:
        xyrange = [-256, 256]
    if zrange is None:
        zrange = [-64, 64]

    if dims ==3:
        xy_origin = np.random.randint(low=xyrange[0], high=xyrange[1], size=(2))
        z_origin = np.random.randint(low=zrange[0],high=zrange[1],size=(1))
        origin = np.concatenate((xy_origin,z_origin))
    else:
        origin=np.random.randint(low=xyrange[0], high=xyrange[1], size=(2))
    return origin

def generate_one_track(step,xyrange=None, zrange=None, dims =3):
    if xyrange is None:
        xyrange = [-256, 256]
    if zrange is None:
        zrange = [-64, 64]
    origin = get_origin()

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
                next_pos = np.array([next_pos[0],current_pos[1]-step_set[1],current_pos[2]])
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


tracks = pd.DataFrame(columns = ['x','y','z','frame','particle_idx'])
for n in tqdm(range(0,n_runs)):
    path = generate_one_track(step_n)
    buffer = pd.DataFrame(columns= ['x','y','z','frame','particle_idx'])
    buffer['x'] = path[:,0]
    buffer['y'] = path[:,1]
    buffer['z'] = path[:,2]
    buffer['frame'] = np.array(range(len(path)))
    buffer['particle_idx'] = [n]*len(path)
    tracks = pd.concat([tracks,buffer],axis=0)

tracks.to_csv('tracks_3D_particle.csv',index=False)

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

# runs = np.arange(n_runs)
# step_shape = (step_n, dims)
# Plot
# fig = plt.figure(figsize=(3, 3), dpi=250)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.grid(False)
# ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
# ax.set_xlabel("X")
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim(-256, 256)
# ax.set_ylim(-256, 256)
# ax.set_zlim(-64, 64)

# path = generate_one_track(10)
# ax.scatter3D(path[:, 0], path[:, 1], path[:, 2],
#              c='b', alpha=0.15, s=1)
# ax.plot3D(path[:, 0], path[:, 1], path[:, 2],
#           c='b', lw=1)

# for i, col in zip(runs, colors):
#     # Simulate steps in 3D
#     origin = get_origin()
#     path = generate_one_track(step_n)
#
#     # Plot the path
#     ax.scatter3D(path[:, 0], path[:, 1], path[:, 2],
#                  c=col, alpha=0.15, s=1)
#     ax.plot3D(path[:, 0], path[:, 1], path[:, 2],
#               c=col, alpha=0.25, lw=0.25)
#     ax.plot3D(start[:, 0], start[:, 1], start[:, 2],
#               c=col, marker="+")
#     ax.plot3D(stop[:, 0], stop[:, 1], stop[:, 2],
#               c=col, marker="o")
#
# plt.title('3D Random Walk - Multiple runs')
# plt.show()