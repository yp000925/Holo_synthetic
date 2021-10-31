import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import trackpy as tp
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#%%
gt = pd.read_csv("3d_tracks_no_overlap.csv")
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
particle_idx = gt['particle_idx']

from itertools import cycle
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for i,(idx,col) in enumerate(zip(particle_idx,colors)):
#      path = gt[gt['particle_idx'] == idx]
#      ax.plot3D(path['x'],path['y'],path['z'],c=col)
# plt.show()
gt_linked = tp.link_df(gt, 50, pos_columns=['x', 'y', 'z'])
tp.plot_traj3d(gt_linked)

result = pd.read_csv("prediction_results_no_overlap.csv")
fig = plt.figure()
linked = tp.link_df(result, 50, pos_columns=['x', 'y', 'z'])
tp.plot_traj3d(linked)


#%%% load gt and prediction
um = 1e-6
mm = 1e-3
cm = 1e-2
nm = 1e-9
depth_range = (1 * cm, 3 * cm)
dep_slice = 256
dep_res = (depth_range[1]-depth_range[0])/dep_slice

gt = pd.read_csv("tracks_3D_particle.csv")
# linked_gt = tp.link_df(gt, 60, pos_columns=['x', 'y', 'z'])
gt['particle'] = gt['particle_idx']
gt['x'] = gt['x']+256
gt['y'] = gt['y']+256
gt['z'] = (gt['z']+64)/128*256*dep_res+depth_range[0]
predict = pd.read_csv("prediction_results.csv")
linked_pred = tp.link_df(predict, 40, pos_columns=['x', 'y', 'z'])
linked_pred['z'] = linked_pred['z']*dep_res+depth_range[0]

#%%
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
tp.plot_traj3d(gt,label=True,ax=ax)
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
tp.plot_traj3d(linked_pred,label=True,ax=ax)

#%% match 1
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
ax.set_title('Ground truth')
picked_gt = gt[gt['particle']==4]
picked_pred = linked_pred[linked_pred['particle']==2]
tp.plot_traj3d(picked_gt, ax=ax)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection ='3d')
ax2.set_title('Prediction')
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
ax2.set_zlim(ax.get_zlim())
tp.plot_traj3d(picked_pred,ax=ax2)

#%% match 2
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
ax.set_title('Ground truth')
picked_gt = gt[gt['particle']==1]
picked_pred = linked_pred[linked_pred['particle']==3]
tp.plot_traj3d(picked_gt, ax=ax)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection ='3d')
ax2.set_title('Prediction')
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
ax2.set_zlim(ax.get_zlim())
tp.plot_traj3d(picked_pred,ax=ax2)

#%% match 3
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
ax.set_title('Ground truth')
picked_gt = gt[gt['particle']==0]
picked_pred = linked_pred[linked_pred['particle']==0]
tp.plot_traj3d(picked_gt, ax=ax)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection ='3d')
ax2.set_title('Prediction')
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
ax2.set_zlim(ax.get_zlim())
tp.plot_traj3d(picked_pred,ax=ax2)


#%%
um = 1e-6
mm = 1e-3
cm = 1e-2
nm = 1e-9
depth_range = (1 * cm, 3 * cm)
dep_slice = 256
dep_res = (depth_range[1]-depth_range[0])/dep_slice

gt = pd.read_csv("tracks_3D_particle2.csv")
gt['particle'] = gt['particle_idx']
gt['x'] = gt['x']+256
gt['y'] = gt['y']+256
# gt['z'] = (gt['z']+64)/128*256*dep_res+depth_range[0]
gt['z'] = (gt['z']+64)/128
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
# ax.set_title('Ground truth')
picked_gt = gt.loc[gt['particle'].isin([4,1,0])]
for p_idx in picked_gt['particle'].unique():
     track = picked_gt[picked_gt['particle']==p_idx].sort_values(['frame'])[0:30]
     ax.plot3D(track['x'],track['y'],track['z'],lw=2,alpha=0.5)
     ax.scatter3D(track['x'].to_numpy()[0], track['y'].to_numpy()[0],track['z'].to_numpy()[0], c='m', marker="^", s=20)
     ax.scatter3D(track['x'].to_numpy()[-1], track['y'].to_numpy()[-1], track['z'].to_numpy()[-1], c='m', marker="o", s=20)

ax.set_xlim([0,512])
ax.set_ylim([0,512])
ax.set_zlim([0,1])
ax.set_xlabel('x(pixels)',fontsize=16)
ax.set_ylabel('y(pixels)',fontsize=16)
ax.set_zlabel('Normalized depth',fontsize=16)

#%%
predict = pd.read_csv("prediction_results.csv")
linked_pred = tp.link_df(predict, 40, pos_columns=['x', 'y', 'z'])
# linked_pred['z'] = linked_pred['z']*dep_res+depth_range[0]
linked_pred['z'] = linked_pred['z']/255.0
fig = plt.figure()
ax2 = fig.add_subplot(111,projection ='3d')
# ax.set_title('Ground truth')
picked_pred = linked_pred.loc[linked_pred['particle'].isin([3,2,0])]
for p_idx in [0,3,2]:
     track = picked_pred[picked_pred['particle']==p_idx].sort_values(['frame'])[0:30]
     ax2.plot3D(track['x'],track['y'],track['z'],lw=2,alpha=0.5)
     ax2.scatter3D(track['x'].to_numpy()[0], track['y'].to_numpy()[0],track['z'].to_numpy()[0], c='m', marker="^", s=20)
     ax2.scatter3D(track['x'].to_numpy()[-1], track['y'].to_numpy()[-1], track['z'].to_numpy()[-1], c='m', marker="o", s=20)

ax2.set_xlim([0,512])
ax2.set_ylim([0,512])
ax2.set_zlim([0,1])
ax2.set_xlabel('x(pixels)',fontsize=16)
ax2.set_ylabel('y(pixels)',fontsize=16)
ax2.set_zlabel('Normalized depth',fontsize=16)

#%%
def change_value(x):
     if x ==0:
          return 0
     if x ==2:
          return 4
     if x == 1 or 5:
          return 2
     if x == 4 or 6:
          return 3
     if x ==3:
          return 1

linked_pred['particle'] = linked_pred['particle'].apply(change_value)



#%% just for plot in the volume
# gt['x'] = gt['x']+256
# gt['y'] = gt['y']+256
# # gt['z'] = (gt['z']+64)/128*256*dep_res+depth_range[0]
# gt['z'] = (gt['z']+64)/128

for_plot = gt.copy()
# for_plot['y'] = gt['z']
# for_plot['z'] = gt['y']
for_plot['particle'] = gt.particle_idx
fig  = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
idx = for_plot.particle.unique()
for p_idx in idx:
     track = for_plot[for_plot['particle'] == p_idx].sort_values(['frame'])
     ax.plot3D(track['x'], track['z'], track['y'], lw=2, alpha=0.5) # for better visualization
     ax.scatter3D(track['x'].to_numpy()[0], track['z'].to_numpy()[0],track['y'].to_numpy()[0], c='m', marker="^", s=20)
     ax.scatter3D(track['x'].to_numpy()[-1], track['z'].to_numpy()[-1], track['y'].to_numpy()[-1], c='m', marker="o", s=20)

# ax.set_xlim([0,512])
# ax.set_zlim([0,512])
# ax.set_ylim([0,1])
ax.set_xlabel('x(pixels)', fontsize=16)
ax.set_zlabel('y(pixels)', fontsize=16)
ax.set_ylabel('Normalized depth', fontsize=16)



#%%
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import interp1d

import numpy as np

def interp_helper(values, num=50, kind='linear'):

     interp_i = np.linspace(min(values), max(values), num)

     return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)

def plot_one_track(ax,x,y,z,kind = 'linear', **kwargs):

     x_new, y_new, z_new = (interp_helper(i, 100, kind=kind) for i in (x, y, z))

     # zmax = np.array(z_new).max()
     # zmin = np.array(z_new).min()

     for i in range(len(z_new) - 1):
          # ax.plot(x_new[i:i + 2], z_new[i:i + 2],y_new[i:i + 2],
          #         color=plt.cm.jet(int(np.array(z_new[i:i + 2]).mean() * 255)), **kwargs)
          ax.plot(x_new[i:i + 2], z_new[i:i + 2], y_new[i:i + 2],
             color=plt.cm.jet(int((i+1)/(len(z_new)+1) * 255)), **kwargs)

def transfer_gt(gt):
     gt['particle'] = gt['particle_idx']
     gt['x'] = gt['x']+256
     gt['y'] = gt['y']+256
# gt['z'] = (gt['z']+64)/128*256*dep_res+depth_range[0]
     gt['z'] = (gt['z']+64)/128
     return gt

tracks = pd.read_csv("3d_track_flow.csv")
gt = transfer_gt(tracks.copy())
for_plot = gt.copy()
for_plot['particle'] = gt.particle_idx
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
idx = for_plot.particle.unique()
for p_idx in idx:
     track = for_plot[for_plot['particle'] == p_idx].sort_values(['frame'])
     # ax.plot3D(track['x'], track['z'], track['y'], lw=2, alpha=0.5) # for better visualization
     plot_one_track(ax, track['x'], track['y'], track['z'])
# ax.set_xlim([0,512])
# ax.set_zlim([0,512])
# ax.set_ylim([0,1])
ax.set_xlabel('x(pixels)', fontsize=16)
ax.set_zlabel('y(pixels)', fontsize=16)
ax.set_ylabel('Normalized depth', fontsize=16)
#
ax.xaxis._axinfo['juggled'] = (2,0,1)
# ax.yaxis._axinfo['juggled'] = (2,1,0)
# ax.zaxis._axinfo['juggled'] = (1,2,0)

predict = pd.read_csv("prediction_results_flow.csv")
linked_pred = tp.link_df(predict, 100, pos_columns=['x', 'y', 'z'])
fig = plt.figure()
ax2 = fig.add_subplot(111, projection ='3d')
idx = for_plot.particle.unique()
for p_idx in idx:
     track = for_plot[for_plot['particle'] == p_idx].sort_values(['frame'])
     # ax.plot3D(track['x'], track['z'], track['y'], lw=2, alpha=0.5) # for better visualization
     plot_one_track(ax2, track['x'], track['y'], track['z'])
# ax.set_xlim([0,512])
# ax.set_zlim([0,512])
# ax.set_ylim([0,1])
ax2.set_xlabel('x(pixels)', fontsize=16)
ax2.set_zlabel('y(pixels)', fontsize=16)
ax2.set_ylabel('Normalized depth', fontsize=16)
#
ax2.xaxis._axinfo['juggled'] = (2,0,1)


import gradient as gd
def plot_results_path(result0_tmp, linewidth_car=4):
    x = result0_tmp[:, 0]
    y = result0_tmp[:, 1]
    z = result0_tmp[:, 2]
    z= None
    # 设置颜色渐变色
    # 'jet', 'cool'
    lc = gd.colorline(x, y, z, cmap=plt.get_cmap('cool'), linewidth=linewidth_car)  # 'jet' #'cool'
    return lc

def plot_gd_bar(fig, ax, lc, max_pro, max_tran=0, cars_num=1, car_num=0, offset=0):
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0 + (car_num / cars_num) * ax.get_position().height,
                        0.02,
                        ax.get_position().height - (
                                    (cars_num - car_num - 1) / cars_num) * ax.get_position().height - offset])
    cb = plt.colorbar(lc, cax=cax)

#%%
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
idx = for_plot.particle.unique()
for p_idx in idx:
     track = for_plot[for_plot['particle'] == p_idx].sort_values(['frame'])
     # ax.plot3D(track['x'], track['z'], track['y'], lw=2, alpha=0.5) # for better visualization
     plot_one_track(ax, track['x'], track['y'], track['z'])
cax = fig.add_axes([ax.get_position().x1 + 0.02,
                    ax.get_position().y0,
                    0.02,
                    ax.get_position().height])

x_new, y_new, z_new = (interp_helper(i, 1000, kind='linear') for i in (track['x'], track['y'], track['z']))

def make_segments(x, y, z):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 3 (x
    and y) array
    """
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

clmap = np.linspace(1,len(z_new),1000)/len(z_new)*255
lc = Line3DCollection(make_segments(x_new, y_new, z_new), array=clmap, cmap=plt.get_cmap('cool'),linewidth=2)
plt.colorbar(lc, cax=cax)

from matplotlib.collections import LineCollection
line_segments = LineCollection(np.array([x_new, y_new]).T, linewidths=(0.5, 1, 1.5, 2),
                                linestyle='solid')

#%%

import pandas as pd
import matplotlib.pyplot as plt


#渐变色

#数据读取


#设置画布
width_img = 5
height_img = 5
fig = plt.figure(figsize=(int(width_img)+2, int(height_img)+2),
                 facecolor='none')
ax = plt.gca()
# #设置图像上下界
# plt.xlim(0,20)
# plt.ylim(0,20)

# #设置字体
# font1 = {'family': 'Times New Roman','weight': 'normal', 'size': 15}
# font2 = {'family': 'Times New Roman','fontstyle': 'italic', 'size': 15}
# plt.tick_params(labelsize = 12)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

#绘制路径结果轨迹
result0 = np.concatenate([x_new[:,None],y_new[:,None],z_new[:,None]],axis=1)
lc = plot_results_path(result0,4)


#label
# plt.xlabel('x [m]', font1)
# plt.ylabel('y [m]', font1)

#绘制渐变色的图例
cb = plot_gd_bar(fig, ax, lc, result0[-1, 2], 10) #最后两个参数一个是调整比例，一个是调整偏移量

plt.show()