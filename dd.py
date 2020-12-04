# %%
import matplotlib.pyplot as plt
import pandas as pd
from LightPipes import *
from propagators import ASM

wavelength = 633 * nm
N = 1024
pixel_pitch = 10 * um
size = 10 * mm  # 10mm * 10mm
size_range = [20 * um, 100 * um]

def generate_holo_fromcsv(file):
    particles_field = pd.read_csv(file)
    grouped = particles_field.groupby(by=['z'])
    F_ini = Begin(size, wavelength, N)
    F_obj_mix = F_ini
    count = 0
    for z, particles in grouped:
        F_ini = Begin(size, wavelength, N)
        F_obj_new = F_ini
        for i in range(len(particles)):
            F_obj_new = CircScreen(F_obj_new, particles.iloc[i]['size'], particles.iloc[i]['x'], particles.iloc[i]['y'])
        F_obj_new = ASM(F_obj_new, z)
        F_obj_mix = BeamMix(F_obj_mix, F_obj_new)
        count+=1
    I = Intensity(F_obj_mix)
    # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
    return I

def generate_holo_fromcsv2(file):
    particles_field = pd.read_csv(file)
    count = 0
    z_list = particles_field['z'].unique()
    F_obj = Begin(size, wavelength, N)
    for i in range(len(z_list)-1):
        prop_dis = z_list[i]-z_list[i+1]
        if prop_dis<0:
            raise ValueError("z is not aceding")
        particles = particles_field[particles_field['z']==z_list[i]]
        for j in range(len(particles)):
            F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
        F_obj = ASM(F_obj, prop_dis)
        count+=1
    F_obj = ASM(F_obj, z_list[-1])
    I = Intensity(F_obj)
    # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
    return I,count


fig1 = plt.imread('Hologram0.png')
fig2 = plt.imread('Hologram4.png')

I1,i1 = generate_holo_fromcsv2("Hologram0.csv")
plt.imsave("I1.png",I1, cmap='gray')
I2,i2 = generate_holo_fromcsv2('Hologram4.csv')
plt.imsave("I2.png", I2, cmap='gray')