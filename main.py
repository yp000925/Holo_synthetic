from LightPipes import *
import matplotlib.pyplot as plt
wavelength = 633*nm
N = 1000
size = 2000*um
particle = 50*um
z1=3*cm

F = Begin(size,wavelength,N)
F=SuperGaussAperture(F,size/3,n=10)
F_obj1 = CircScreen(F,particle)

F_obj1Fo = Forvard(F_obj1,z1)

F_obj1Fr = Fresnel(F_obj1,z1)


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

ax1.imshow(Intensity(F_obj1Fo,2), cmap='gray'); ax1.axis('off');ax1.set_title('Forvard')
ax2.imshow(Intensity(F_obj1Fr,2), cmap='gray'); ax2.axis('off');ax2.set_title('Fresnel')

plt.show()