import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import trackpy as tp
import pandas as pd
import numpy as np
import gradient as gd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_results_path(result0_tmp, linewidth_car=4):
    x = result0_tmp[:, 0]
    y = result0_tmp[:, 1]
    z = result0_tmp[:, 2]

    # 设置颜色渐变色
    # 'jet', 'cool'
    lc = gd.colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=linewidth_car)  # 'jet' #'cool'
    return lc

def plot_gd_bar(fig, ax, lc, max_pro, max_tran=0, cars_num=1, car_num=0, offset=0):
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0 + (car_num / cars_num) * ax.get_position().height,
                        0.02,
                        ax.get_position().height - (
                                    (cars_num - car_num - 1) / cars_num) * ax.get_position().height - offset])
    cb = plt.colorbar(lc, cax=cax)


def interp_helper(values, num=50, kind='linear'):
    interp_i = np.linspace(min(values), max(values), num)

    return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)


def transfer_gt(gt):
    gt['particle'] = gt['particle_idx']
    gt['x'] = gt['x'] + 256
    gt['y'] = gt['y'] + 256
    # gt['z'] = (gt['z']+64)/128*256*dep_res+depth_range[0]
    gt['z'] = (gt['z'] + 64) / 128
    return gt

def make_segments_3d(x, y, z):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 3 (x
    and y) array
    """
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def get_linecollection(x,y,z):
    x_new, y_new, z_new = (interp_helper(i, 1000, kind='linear') for i in (x,y,z))
    segments = make_segments_3d(x_new, y_new, z_new)
    clmap = np.linspace(1, len(z_new), segments.shape[0]) / len(z_new)
    lc = Line3DCollection(segments, array=clmap, cmap=plt.get_cmap('jet'), linewidth=2)
    return lc

tracks = pd.read_csv("3d_track_flow.csv")
gt = transfer_gt(tracks.copy())
for_plot = gt.copy()
idx = for_plot.particle.unique()
fig = plt.figure() 
ax = Axes3D(fig,azim=60,elev=10)

for p_idx in idx:
     track = for_plot[for_plot['particle'] == p_idx].sort_values(['frame'])
     lc = get_linecollection(track['x'],track['z'], track['y'])
     ax.add_collection(lc)


#设置画布
# width_img = 5
# height_img = 5
# fig = plt.figure(figsize=(int(width_img)+2, int(height_img)+2),
#                  facecolor='none')

# #设置字体
# font1 = {'family': 'Times New Roman','weight': 'normal', 'size': 15}
# font2 = {'family': 'Times New Roman','fontstyle': 'italic', 'size': 15}
# plt.tick_params(labelsize = 12)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

ax.set_position([0, 0.0, 1.0, 1.0])
# ax.add_collection(lc)
plt.tick_params(labelsize = 18)
ax.set_xlim(0,512)
ax.set_ylim(0,1)
ax.set_zlim(0,512)
ax.invert_xaxis()
ax.invert_zaxis()
ax.xaxis._axinfo['juggled'] = (2,0,1)
#label
# plt.xlabel('x [m]', font1)
# plt.ylabel('y [m]', font1)

#绘制渐变色的图例
ax = plt.gca()
cax = fig.add_axes([ax.get_position().x1 + 0.02,
                    ax.get_position().y0 + 0.05,
                    0.02,
                    ax.get_position().height-0.15])
plt.colorbar(lc, cax=cax)
cax.set_yticklabels([''])
# plt.show()

#%% plot prediction
predict = pd.read_csv("prediction_results_flow.csv")
linked_pred = tp.link_df(predict, 100, pos_columns=['x', 'y', 'z'])
linked_pred['z'] = linked_pred['z']/256.0
idx = linked_pred.particle.unique()
fig2 = plt.figure()
ax2 = Axes3D(fig2,azim=60,elev=10)
for p_idx in idx:
     track = linked_pred[linked_pred['particle'] == p_idx].sort_values(['frame'])
     lc2 = get_linecollection(track['x'],track['z'], track['y'])
     ax2.add_collection(lc2)

ax2.set_position([0, 0.0, 1.0, 1.0])
# ax.add_collection(lc)
plt.tick_params(labelsize = 18)
ax2.set_xlim(0,512)
ax2.set_ylim(0,1)
ax2.set_zlim(0,512)
ax2.invert_xaxis()
ax2.invert_zaxis()
ax2.xaxis._axinfo['juggled'] = (2,0,1)
#label
# plt.xlabel('x [m]', font1)
# plt.ylabel('y [m]', font1)

#绘制渐变色的图例
ax2 = plt.gca()
cax2 = fig2.add_axes([ax2.get_position().x1 + 0.02,
                    ax2.get_position().y0 + 0.05,
                    0.02,
                    ax2.get_position().height-0.15])
plt.colorbar(lc2, cax=cax2)
cax2.set_yticklabels([''])