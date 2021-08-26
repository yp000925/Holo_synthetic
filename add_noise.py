from PIL import Image
import numpy as np

image_path = '7650.png'
img = Image.open(image_path).convert('L')
img = np.array(img)

def add_gaussian_noise(img_in,snr):
    temp  = np.float64(np.copy(img_in))
    h,w = temp.shape
    p_img = np.mean(np.power(img_in,2))
    # noise_sigma = (np.power(10,snr/10))
    noise_sigma = p_img / (np.power(10,10/snr))
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp.shape, np.float64)
    if len(temp.shape) == 2:
        noisy_image = temp+noise
    else:
        noisy_image[:, :, 0] = temp[:, :, 0] + noise
        noisy_image[:, :, 1] = temp[:, :, 1] + noise
        noisy_image[:, :, 2] = temp[:, :, 2] + noise
    return np.clip(noisy_image, a_min=0, a_max=255)


snr = 0.1
n_img = add_gaussian_noise(img, snr)
n_img = Image.fromarray(n_img)
n_img.show()