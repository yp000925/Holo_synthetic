from LightPipes import *
from utils import *
import time
# from LightPipes.propagators import ASM
from propagators import ASM
# using Fresnel propagation with ASM

wavelength = 633*nm
N = 1024
pixel_pitch = 10*um
size = pixel_pitch*N # 10mm * 10mm
particle = 50*um/2

[x1,y1,z1] = [0,0,3*cm]
[x2,y2,z2] = [-3.0*mm,1.2*mm,3*cm]
[x3,y3,z3] = [0.3*mm,0.3*mm,1*cm]


f_factor1 = 2*particle**2/(wavelength*z1)
f_factor2 = 2*particle**2/(wavelength*z2)
f_factor3 = 2*particle**2/(wavelength*z3)

#%%
F = Begin(size,wavelength,N)
F_obj1 = CircScreen(F,particle,x1,y1)
F_obj2 = CircScreen(F,particle,x2,y2)
F_obj3 = CircScreen(F,particle,x3,y3)

t1 = time.time()
F_obj1 = ASM(F_obj1,z1)
t2 = time.time()
print('The time for compute one propagation is {:.2f} s'.format(t2-t1))
F_obj2 = ASM(F_obj2,z2)
F_obj3 = ASM(F_obj3,z3)
Mix = BeamMix(F_obj1,F_obj2)
Mix = BeamMix(Mix,F_obj3)

I = Intensity(Mix) #255
plt.imshow(I,cmap='gray');plt.axis('off');plt.title("Spectrum method FFT+IFFT");plt.show()
plt.imsave("ImageASM_1.png",I,cmap='gray')

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(Intensity(Mix,2), cmap='gray'); ax1.axis('off');ax1.set_title('Mix')
ax2.imshow(Intensity(F_obj1,2), cmap='gray'); ax2.axis('off');ax2.set_title('First particle at z=%f'%z1)
ax3.imshow(Intensity(F_obj2,2), cmap='gray'); ax3.axis('off');ax3.set_title('Second particlez z=%f'%z2)
ax4.imshow(Intensity(F_obj3,2), cmap='gray'); ax4.axis('off');ax4.set_title('Third particle z=%f'%z3)
plt.show()
fig.savefig("ImageASM_2.png")


#%%
size_extend = size*5
N_extend = N*5
F = Begin(size_extend,wavelength,N_extend)
F_obj1 = CircScreen(F,particle,x1,y1)
F_obj2 = CircScreen(F,particle,x2,y2)
F_obj3 = CircScreen(F,particle,x3,y3)

t1 = time.time()
F_obj1 = ASM(F_obj1,z1)
t2 = time.time()
print('The time for compute one propagation is {:.2f} s'.format(t2-t1))
F_obj2 = ASM(F_obj2,z2)
F_obj3 = ASM(F_obj3,z3)
Mix = BeamMix(F_obj1,F_obj2)
Mix = BeamMix(Mix,F_obj3)

I = Intensity(Mix)

I_crop = crop_intensity(I,N)
plt.imshow(I_crop,cmap='gray');plt.axis('off');plt.title("Spectrum method by extending field");plt.show()
plt.imsave("ImageASME_1.png",I_crop,cmap='gray')

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(crop_intensity(Intensity(Mix),N), cmap='gray'); ax1.axis('off');ax1.set_title('Mix by extending field')
ax2.imshow(crop_intensity(Intensity(F_obj1),N), cmap='gray'); ax2.axis('off');ax2.set_title('First particle at z=%f'%z1)
ax3.imshow(crop_intensity(Intensity(F_obj2),N), cmap='gray'); ax3.axis('off');ax3.set_title('Second particlez z=%f'%z2)
ax4.imshow(crop_intensity(Intensity(F_obj3),N), cmap='gray'); ax4.axis('off');ax4.set_title('Third particle z=%f'%z3)
plt.show()
fig.savefig("ImageASME_2.png")