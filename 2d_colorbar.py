import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import trackpy as tp
import pandas as pd
import numpy as np
import gradient as gd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
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




tracks = pd.read_csv("3d_track_flow.csv")
gt = transfer_gt(tracks.copy())
for_plot = gt.copy()
idx = for_plot.particle.unique()
for p_idx in idx:
     track = for_plot[for_plot['particle'] == p_idx].sort_values(['frame'])
     break

x = np.linspace(0,1,100)
y = x
# x_new, y_new, z_new = (interp_helper(i, 1000, kind='linear') for i in (track['x'], track['y'], track['z']))
# x_new = x_new/x_new.max()
# y_new = y_new/y_new.max()
# z_new = z_new/z_new.max()
x_new, y_new, z_new = (interp_helper(i, 1000, kind='linear') for i in (x,y,x))
# #渐变色
#
# #数据读取


#设置画布
# width_img = 5
# height_img = 5
# fig = plt.figure(figsize=(int(width_img)+2, int(height_img)+2),
#                  facecolor='none')

ax = plt.gca()
#设置图像上下界
plt.xlim(0,x_new.max())
plt.ylim(0,y_new.max())


# #设置字体
# font1 = {'family': 'Times New Roman','weight': 'normal', 'size': 15}
# font2 = {'family': 'Times New Roman','fontstyle': 'italic', 'size': 15}
# plt.tick_params(labelsize = 12)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

#绘制路径结果轨迹
result0 = np.concatenate([x_new[:,None],y_new[:,None],z_new[:,None]], axis=1)
lc = plot_results_path(result0,4)


#label
# plt.xlabel('x [m]', font1)
# plt.ylabel('y [m]', font1)

#绘制渐变色的图例


cb = plot_gd_bar(fig, ax, lc, result0[-1, 2], 10) #最后两个参数一个是调整比例，一个是调整偏移量

plt.show()