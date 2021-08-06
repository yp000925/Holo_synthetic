from LightPipes import *
import matplotlib.pyplot as plt

def Particle(Fin, x_shift=0.0, y_shift=0.0):
    """
    *Inserts a circular screen in the field.*

    :param Fin: input field
    :type Fin: Field
    :param R: radius of the screen
    :type R: int, float
    :param x_shift: shift in x direction (default = 0.0)
    :param y_shift: shift in y direction (default = 0.0)
    :type x_shift: int, float
    :type y_shift: int, float
    :return: output field (N x N square array of complex numbers).
    :rtype: `LightPipes.field.Field`
    :Example:
    .. seealso::

        * :ref:`Manual: Apertures and screens<Apertures and screens.>`

        * :ref:`Examples: Spot of Poisson <Spot of Poisson.>`
    """
    # from
    # https://stackoverflow.com/questions/44865023/
    # circular-masking-an-image-in-python-using-numpy-arrays
    Fout = Fin
    Y, X = Fout.mgrid_cartesian
    Y = Y - y_shift
    X = X - x_shift
    dist_sq = X ** 2 + Y ** 2  # squared, no need for sqrt
    step = Fout.grid_step
    Fout.field[dist_sq < step ** 2] = 0.0
    return Fout

def crop_intensity(field,cropped_N):
    N = field.shape[0]
    return field[int(N/2-cropped_N/2):int(N/2+cropped_N/2),int(N/2-cropped_N/2):int(N/2+cropped_N/2)]

if __name__ == '__main__':
    wavelength = 633 * nm
    N = 128
    pixel_pitch = 10 * um
    size = pixel_pitch * N
    particle = pixel_pitch
    z = 2 * mm
    F_obj = Begin(size, wavelength, N)
    F_obj = Particle(F_obj)
    I = Intensity(F_obj)
    plt.imshow(I, cmap='gray')
    plt.show()