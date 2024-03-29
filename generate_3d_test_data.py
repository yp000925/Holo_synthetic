import pandas as pd

from utils import *
from propagators import ASM
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

um = 1e-6
mm = 1e-3
cm = 1e-2
nm = 1e-9

size_range = [50 * um, 50 * um]
depth_range = (1 * cm, 3 * cm)
N = 512
pixel_pitch = 10 * um
frame_size = pixel_pitch * N
dep_slice = 256
dep_res = (depth_range[1]-depth_range[0])/dep_slice

file = 'tracks_3D_particle.csv'

# file = 'test.csv'
f = pd.read_csv(file)


import trackpy as tp



f['x'] = f['x']*pixel_pitch
f['y'] = f['y']*pixel_pitch
f['z'] = (f['z']+64)*dep_res*2+depth_range[0]



wavelength = 633 * nm
size_range = [50*um,50*um]

def get_size(smin,smax,number):
    s = (smax-smin)*np.random.random(number)+smin
    return s

def generate_holo_fromfield(particles_field):
    particles_field = particles_field.sort_values(by=['z'], ascending=False)
    s = get_size(size_range[0], size_range[1], len(particles_field))
    particles_field['size'] = s

    z_list = particles_field['z'].unique()
    F_obj = Begin(frame_size, wavelength, N)
    for i in range(len(z_list)-1):
        prop_dis = z_list[i]-z_list[i+1]
        if prop_dis < 0:
            raise ValueError("z is not aceding")
        particles = particles_field[particles_field['z']==z_list[i]]
        for j in range(len(particles)):
            F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
        F_obj = ASM(F_obj, prop_dis)

    particles = particles_field[particles_field['z'] == z_list[-1]]
    for j in range(len(particles)):
        F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
    F_obj = ASM(F_obj, z_list[-1])
    I = Intensity(F_obj)
    # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
    return I

# fig = plt.figure(figsize=(3, 3), dpi=250)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.grid(False)
# particle_idx = f['particle_idx']
# for idx in particle_idx:
#     tracks = f[f['particle_idx'] == idx]
#     # ax.plot3D(tracks['x'], tracks['y'], tracks['z'],
#     #           c='b', lw=0.25)
#
#

frames = f['frame']
from tqdm import tqdm
for frame_idx in tqdm(frames):
    particles = f[f.frame==frame_idx]
    holo = generate_holo_fromfield(particles)
    img = Image.fromarray((holo / np.max(holo) * 255).astype(np.uint8)).convert('RGB')
    img.show()
    name = 'frame%d'%frame_idx+'.png'
    img.save('3dtrack/'+name)
    if frame_idx == 5:
        break
