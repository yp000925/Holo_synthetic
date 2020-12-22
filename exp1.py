#check the bbox range

from LightPipes import *
from utils import *
from propagators import ASM
import numpy as np
import  pandas as pd
import  time

wavelength = 633 * nm
N = 1024
# pixel_pitch = 10*um
size = 10 * mm  # 10mm * 10mm
size_range = [20 * um, 100 * um]
depth_range = [1*cm, 3*cm]

F = Begin(size,wavelength,N)
F = CircScreen(F,size_range[0],2.5*mm,2.5*mm)
F = CircScreen(F,size_range[1],-2.5*mm,2.5*mm)
F = ASM(F,2*cm)
F = CircScreen(F,size_range[0],-2.5*mm,-2.5*mm)
F = CircScreen(F,size_range[1],2.5*mm,-2.5*mm)
F = ASM(F,1*cm)
I = Intensity(F)
# plt.imshow(I,cmap='gray')
plt.imsave("test.png",I,cmap='gray')

frame = size
def get_buffer(z,size):
    z_rate = (z-1*cm)/(3*cm-1*cm)
    size_rate = (size-size_range[0])/(size_range[1]-size_range[0])
    p_size = int(size_range[1]/frame * N)
    buffer = p_size*10*(z_rate*0.6+size_rate*0.4)
    return buffer

def get_bbox(x,y,z,size):
    px = int(x/frame*N+N/2)
    py = int(N/2+y/frame*N)
    # p_size = int(size/frame*N)
    # buffer = p_size*10
    buffer = get_buffer(z,size)
    bbox_x = max(0, px-buffer)
    bbox_y = max(0, py-buffer)
    height = buffer*2
    width = buffer*2
    if bbox_x+width > N:
        width = N-bbox_x
    if bbox_y+height > N:
        height = N-bbox_y
    seg = [bbox_x,bbox_y,bbox_x,bbox_y+height,bbox_x+width,bbox_y+height,bbox_x+width,bbox_y]
    return (bbox_x,bbox_y,width,height,seg)

px = int(2.5*mm/size*N+N/2)
py = int(N/2+2.5*mm/size*N)
p_size = int(size_range[1]/frame * N)


px = int(-2.5*mm/size*N+N/2)
py = int(N/2-2.5*mm/size*N)
