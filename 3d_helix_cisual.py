#%%
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import trackpy as tp
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


um = 1e-6
mm = 1e-3
cm = 1e-2
nm = 1e-9
depth_range = (1 * cm, 3 * cm)
dep_slice = 256
dep_res = (depth_range[1]-depth_range[0])/dep_slice

gt = pd.read_csv("helix_3D.csv")
# linked_gt = tp.link_df(gt, 60, pos_columns=['x', 'y', 'z'])
gt['particle'] = gt['particle_idx']
gt['x'] = gt['x']+256
gt['y'] = gt['y']+256
# gt['z'] = (gt['z'])*dep_res+depth_range[0]
# gt['z'] = (gt['z']/256*100).astype(np.int)/100
predict = pd.read_csv("prediction_results_helix.csv")
linked_pred = tp.link_df(predict, 40, pos_columns=['x', 'y', 'z'])
# linked_pred['z'] = linked_pred['z']*dep_res+depth_range[0]
# linked_pred['z'] =(linked_pred['z']/256*100).astype(np.int)/100
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
tp.plot_traj3d(gt,ax=ax,plot_style={'linewidth':2})
# ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
# ax.set_yticklabels(ax.get_yticklabels(),fontsize=12)
# ax.set_title('True trajectory',fontsize=20)
ax.set_xlabel('x(pixels)',fontsize=16)
ax.set_ylabel('y(pixels)',fontsize=16)
ax.set_zlabel('Normalized depth',fontsize=16)
#%%
ax.set_zticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=12)
ax.scatter3D(gt.iloc[0].x, gt.iloc[0].y, gt.iloc[0].z, c='m', marker="^", s=50)
ax.scatter3D(gt.iloc[-1].x, gt.iloc[-1].y, gt.iloc[-1].z, c='m', marker="o", s=50)


fig = plt.figure()
ax2 = fig.add_subplot(111,projection ='3d')
tp.plot_traj3d(linked_pred,ax=ax2,plot_style={'linewidth':2})
# ax2.set_title('Prediction')
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
ax2.set_zlim(ax.get_zlim())
ax2.set_xlabel('x(pixels)',fontsize=16)
ax2.set_ylabel('y(pixels)',fontsize=16)
ax2.set_zlabel('Normalized depth',fontsize=16)
ax2.set_zticklabels(['0','0.2','0.4','0.6','0.8','1.0'],fontsize=12)
ax2.scatter3D(linked_pred.iloc[0].x, linked_pred.iloc[0].y, linked_pred.iloc[0].z, c='m', marker="^", s=50)
ax2.scatter3D(linked_pred.iloc[-1].x, linked_pred.iloc[-1].y, linked_pred.iloc[-1].z, c='m', marker="o", s=50)
ax.set_zlim([0,260])
ax2.set_zlim([0,260])