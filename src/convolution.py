import matplotlib.pyplot as plt
import numpy as np
import skimage as sk


def convolve2d(img, kernel, pad_val=0, logging=False):
    result = np.zeros_like(img)
    krn_height, krn_width = kernel.shape
    pad_height, pad_width = krn_height // 2, krn_width // 2
    result_animation = []

    if len(img.shape) == 2:
        padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=pad_val)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum(padded_image[i:i+krn_height, j:j+krn_width] * kernel)
                result_animation.append(result)

    elif len(img.shape) == 3:
        for ch in range(result.shape[2]):
            if logging: print("Started convolution on layer ", ch+1, ".")
            ch_kernel = kernel
            if ch == 1 and np.max(img[:, :, ch]) > 1:
                ch_kernel = kernel / 2

            if logging: progress = 0
            padded_channel = np.pad(img[:, :, ch], ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=pad_val)
            for i in range(result.shape[0]):
                if logging:
                    progress_new = int(((ch/result.shape[2]) + (i/result.shape[0])/(result.shape[2])) * 100)
                    if progress != progress_new:
                        progress = progress_new
                        print(progress, "% complete.")
                for j in range(result.shape[1]):
                    result[i, j, ch] = np.clip(np.sum(padded_channel[i:i+krn_height, j:j+krn_width] * ch_kernel), 0, 255)
                    result_animation.append(result)

    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img

    return result, result_animation


def normalize_image(img):
    if len(img.shape) == 2:
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > 1:
            return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            print("Normalization for float values..." + "\n" +
                  "Min: " + str(min_val) + "\n" +
                  "Max: " + str(max_val) + "\n")
            print("Normalization max: " + str(np.max((img - min_val) / (max_val - min_val))) + "\n" +
                  "Normalization min: " + str(np.min((img - min_val) / (max_val - min_val))) + "\n")
            return ((img - min_val) / (max_val - min_val)).astype(np.float32)
    elif len(img.shape) == 3:
        for ch in range(img.shape[2]):
            min_val = np.min(img[:, :, ch])
            max_val = np.max(img[:, :, ch])
            if max_val > 1:
                img[:, :, ch] = ((img[:, :, ch] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                img[:, :, ch] = ((img[:, :, ch] - min_val) / (max_val - min_val)).astype(np.float32)

        return img
    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img
