import numpy as np

import src.interpolation as interp
from src.convolution import convolve_img, normalize_image
from src.extractions import extract_red, extract_green, extract_blue
from src.utilities import print_channels_data


def demosaic_bayer_interp(img) -> np.ndarray:
    """
    Demosaic the image using bi-linear interpolation
    :param img: Image to demosaic -> ndarray[H, W, 3]
    :return: Demosaiced image -> ndarray[H, W, 3]
    """
    # Separate color channels
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        red = extract_red(img)
        green = extract_green(img)
        blue = extract_blue(img)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img

    # Interpolate channels
    red = interp.interpolate_red(red)
    green = interp.interpolate_green(green)
    blue = interp.interpolate_blue(blue)

    print_channels_data(red, green, blue)  # Print data about the channels

    result_img = np.dstack([red, green, blue])  # Merge channels
    print("Result image: " + str(result_img.shape))  # Print data about the result image

    return result_img


def demosaic_bayer_conv(img, kernel):
    """
    Demosaic the image using convolution
    :param img: Image to demosaic -> ndarray[H, W, 3], ndarray[H, W]
    :param kernel: Kernel array for convolution
    :return: Demosaiced image -> ndarray[H, W, 3]
    """
    result_img = convolve_img(img=img, kernel=kernel)

    # Normalize values to the valid range (0-255)
    result_img = normalize_image(result_img)

    return result_img
