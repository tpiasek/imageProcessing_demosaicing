import matplotlib.pyplot as plt
import numpy as np
import skimage as sk


def convolve_img(img, kernel, padding=0):
    """
    Apply padding onto an image and convolve it with a given kernel.
    :param img: image to be convolved
    :param kernel: kernel to be convolved
    :param padding: padding to be applied
    :return: convolved image
    """
    result = np.zeros_like(img)
    krn_height, krn_width = kernel.shape
    padding_height, padding_width = krn_height // 2, krn_width // 2

    if len(img.shape) == 2: # Convolution for 2D images (grayscale)
        padded_image = np.pad(img, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant', constant_values=padding)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum(padded_image[i:i+krn_height, j:j+krn_width] * kernel)

    elif len(img.shape) == 3: # Convolution for 3D images (color)
        for channel in range(result.shape[2]):
            channel_kernel = kernel
            if channel == 1 and np.max(img[:, :, channel]) > 1:
                channel_kernel = kernel / 2

            padded_channel = np.pad(img[:, :, channel], ((padding_height, padding_height), (padding_width, padding_width)), mode='constant', constant_values=padding)
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i, j, channel] = np.clip(np.sum(padded_channel[i:i+krn_height, j:j+krn_width] * channel_kernel), 0, 255)

    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img

    return result


def normalize_image(img):
    if len(img.shape) == 2:
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > 1:  # integers
            return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:  # floats
            return ((img - min_val) / (max_val - min_val)).astype(np.float32)
    elif len(img.shape) == 3:
        for channel in range(img.shape[2]):
            min_val = np.min(img[:, :, channel])
            max_val = np.max(img[:, :, channel])
            if max_val > 1:  # integers
                img[:, :, channel] = ((img[:, :, channel] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:  # floats
                img[:, :, channel] = ((img[:, :, channel] - min_val) / (max_val - min_val)).astype(np.float32)

        return img
    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img
