from LightPipes import *
from utils import *
from propagators import ASM
import numpy as np
import  pandas as pd
import  time
from PIL import Image

def get_xylocation(size,number):
    x = np.random.random(number)*size-float(size/2)
    y = np.random.random(number)*size-float(size/2)
    return x, y

def get_zlocation(z_list,number):
    z = np.random.choice(z_list,number,replace=True)
    return z

def get_size(smin,smax,number):
    s = (smax-smin)*np.random.random(number)+smin
    return s

def particle_field(number,xyrange,z_list,size_range):
    df = pd.DataFrame()
    x,y = get_xylocation(xyrange,number)
    z = get_zlocation(z_list,number)
    s = get_size(size_range[0],size_range[1],number)
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df['size'] = s
    return df

# 缺少最后 z_list[-1] 的particles
# def generate_holo_fromcsv2(file):
#     particles_field = pd.read_csv(file)
#     z_list = particles_field['z'].unique()
#     F_obj = Begin(size, wavelength, N)
#     for i in range(len(z_list)-1):
#         prop_dis = z_list[i]-z_list[i+1]
#         if prop_dis<0:
#             raise ValueError("z is not aceding")
#         particles = particles_field[particles_field['z']==z_list[i]]
#         for j in range(len(particles)):
#             F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
#         F_obj = ASM(F_obj, prop_dis)
#     F_obj = ASM(F_obj, z_list[-1])
#     I = Intensity(F_obj)
#     # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
#     return I

def generate_holo_fromcsv2(file):
    particles_field = pd.read_csv(file)
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

    particles = particles_field[particles_field['z'] == z_list[-1]]
    for j in range(len(particles)):
        F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
    F_obj = ASM(F_obj, z_list[-1])
    I = Intensity(F_obj)
    # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
    return I


if __name__ == '__main__':
    wavelength = 633*nm
    N = 1024
    # pixel_pitch = 10*um
    size = 10*mm # 10mm * 10mm
    size_range = [50*um,50*um] # radii
    t1 = time.time()
    # generate the random 3D location
    # NUMBER = np.random.randint(low=20, high=50, dtype=int)
    NUMBER = 10
    # Z_list = np.array(np.linspace(1*cm, 3*cm, 256))
    Z_list = np.array([100*mm, 120*mm, 140*mm])
    particles = particle_field(NUMBER,xyrange=8.5*mm,z_list=Z_list,size_range=size_range)
    particles = particles.sort_values(by=['z'],ascending=False)
    particles.to_csv("/Users/zhangyunping/PycharmProjects/Holo_preprocess/test_particles2.csv",index=False)
    holo = generate_holo_fromcsv2("/Users/zhangyunping/PycharmProjects/Holo_preprocess/test_particles2.csv")
    plt.imsave("/Users/zhangyunping/PycharmProjects/Holo_preprocess/test_particles2.png", holo, cmap='gray')






