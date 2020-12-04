from LightPipes import *
from utils import *
from propagators import ASM
import numpy as np
import  pandas as pd
import  time

def get_xylocation(size,number):
    x = np.random.random(number)*size-float(size/2)
    y = np.random.random(number)*size-float(size/2)
    return x,y

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
    F_obj = ASM(F_obj, z_list[-1])
    I = Intensity(F_obj)
    # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
    return I

if __name__ == '__main__':
    wavelength = 633*nm
    N = 1024
    pixel_pitch = 10*um
    size = 10*mm # 10mm * 10mm
    size_range = [20*um,100*um]

    for n in range(10):
        t1 = time.time()
        # generate the random 3D location
        NUMBER = np.random.randint(low=50,high=200,dtype=int)
        Z_list = np.array(np.linspace(1*cm, 3*cm, 256))

        particles = particle_field(NUMBER,size,Z_list,size_range=size_range)
        particles = particles.sort_values(by=['z'],ascending=False)
        particles.to_csv("Hologram%d.csv"% n,index=False)

        particles_field = pd.read_csv("Hologram%d.csv"% n)
        grouped = particles_field.groupby(by=['z'])
        F_ini = Begin(size, wavelength, N)
        F_obj_mix = F_ini
        for z, particles in grouped:
            F_ini = Begin(size, wavelength, N)
            for i in range(len(particles)):
                F_obj_new = CircScreen(F_ini, particles.iloc[i]['size'], particles.iloc[i]['x'], particles.iloc[i]['y'])
            F_obj_new = ASM(F_obj_new,z)
            F_obj_mix = BeamMix(F_obj_mix,F_obj_new)
        I = Intensity(F_obj_mix,2)
        plt.imsave("Hologram%d.png"% n, I, cmap='gray')
        t2 = time.time()
        print('%5d / 1000 Time for hologram %f s with %d particle'% (n,(t2-t1),NUMBER))






