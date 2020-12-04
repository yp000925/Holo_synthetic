from LightPipes import *
from utils import *
_USE_PYFFTW = True
_using_pyfftw = False # determined if loading is successful
if _USE_PYFFTW:
    try:
        import pyfftw as _pyfftw
        from pyfftw.interfaces.numpy_fft import fft2 as _fft2
        from pyfftw.interfaces.numpy_fft import ifft2 as _ifft2
        from pyfftw.interfaces.numpy_fft import fftshift as _fftshift
        from pyfftw.interfaces.numpy_fft import ifftshift as _ifftshift
        _fftargs = {'planner_effort': 'FFTW_ESTIMATE',
                    'overwrite_input': True,
                    'threads': -1} #<0 means use multiprocessing.cpu_count()
        _using_pyfftw = True
    except ImportError:
        import warnings
        warnings.warn('LightPipes: Cannot import pyfftw,'
                      + ' falling back to numpy.fft')
if not _using_pyfftw:
    from numpy.fft import fft2 as _fft2
    from numpy.fft import ifft2 as _ifft2
    from pyfftw.interfaces.numpy_fft import fftshift as _fftshift
    from pyfftw.interfaces.numpy_fft import ifftshift as _ifftshift
    _fftargs = {}
import numpy as _np

def ASM(Fin, z):
    # Fout = Field.shallowcopy(Fin)
    Fout = Fin
    wavelength = Fout.lam
    N = Fout.N
    size = Fout.siz
    # deltaX = size/N
    # deltaY = size/N
    k = 2 * _np.pi / wavelength
    a1 = Fout.field
    A1 = _fftshift(_fft2(_fftshift(a1)))
    r1 = _np.linspace(-a1.shape[0] / 2, a1.shape[1] / 2 - 1, a1.shape[0])
    s1 = _np.linspace(-a1.shape[0] / 2, a1.shape[1] / 2 - 1, a1.shape[0])
    deltaFX = 1 / size * r1
    deltaFY = 1 / size * s1
    meshgrid = _np.meshgrid(deltaFX, deltaFY)
    H = _np.exp(
        1.0j * k * z * _np.sqrt(1 - _np.power(wavelength * meshgrid[0], 2) - _np.power(wavelength * meshgrid[1], 2)))
    U = _np.multiply(A1, H)
    u = _ifftshift(_ifft2(_ifftshift(U)))
    Fout.field = u
    return Fout

# TODO: write the bilinear interpolation, try to interpolate the result after FFT then IFFT to see whether aliasing still occurs
def ASM_Ext_fail(Fin,z,factor = 5):
    wavelength = Fin.lam
    N = Fin.N
    size = Fin.siz
    deltax = size/N
    deltay = size/N
    F_ext = Interpol(Fin,size,N*factor)
    k = 2 * _np.pi / wavelength
    a1 = F_ext.field
    A1 = _fftshift(_fft2(_fftshift(a1)))
    r1 = _np.linspace(-a1.shape[0] / 2, a1.shape[1] / 2 - 1, a1.shape[0])
    s1 = _np.linspace(-a1.shape[0] / 2, a1.shape[1] / 2 - 1, a1.shape[0])
    deltaFX = 1 / (N*deltax) * r1
    deltaFY = 1 / (N*deltay) * s1
    meshgrid = _np.meshgrid(deltaFX, deltaFY)
    H = _np.exp(
        1.0j * k * z * _np.sqrt(1 - _np.power(wavelength * meshgrid[0], 2) - _np.power(wavelength * meshgrid[1], 2)))
    U = _np.multiply(A1, H)
    u = _ifftshift(_ifft2(_ifftshift(U)))
    Fout = Fin
    Fout.field = crop_intensity(u,N)
    return Fout

def ASM_crop(Fin,z,factor = 5):
    wavelength = Fin.lam
    # N = Fin.N*factor
    size = Fin.siz
    k = 2 * _np.pi / wavelength
    a1 = Fin.field
    A1 = _fftshift(_fft2(_fftshift(a1)))
    r1 = _np.linspace(-a1.shape[0] / 2, a1.shape[1] / 2 - 1, a1.shape[0])
    s1 = _np.linspace(-a1.shape[0] / 2, a1.shape[1] / 2 - 1, a1.shape[0])

    deltaFX = 1 / size * r1
    deltaFY = 1 / size * s1
    meshgrid = _np.meshgrid(deltaFX, deltaFY)
    H = _np.exp(
        1.0j * k * z * _np.sqrt(1 - _np.power(wavelength * meshgrid[0], 2) - _np.power(wavelength * meshgrid[1], 2)))
    U = _np.multiply(A1, H)
    u = _ifftshift(_ifft2(_ifftshift(U)))
    Fout = Fin
    Fout.field = crop_intensity(u,N)
    return Fout

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wavelength = 633 * nm
    N = 1024

    particle = 50 * um/2
    z = 3 * cm
    x_shift = 0

    pixel_pitch = 10 * um
    size = N * pixel_pitch
    F_obj = Begin(size, wavelength, N)
    F_obj = CircScreen(F_obj, particle)
    F_obj = ASM(F_obj, z)
    I = Intensity(F_obj)
    plt.imshow(I, cmap='gray');
    plt.axis('off');
    plt.title("Spectrum method by extending field");
    plt.show()







