import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from PIL.ImageColor import colormap
from networkx.algorithms.bipartite.basic import color


def split_into_squares(arr, square_size):
    h, w = arr.shape  # Get height and width of the array

    while w % square_size != 0 and square_size > 0:
        square_size -= 1

    if square_size <= 1:
        raise ValueError("Square size must be greater than 1.")

    no_squares_w = w // square_size + (1 if w%square_size>0 else 0)
    no_squares_h = h // square_size + (1 if h%square_size>0 else 0)

    # Calculate dimensions of the padded array (we add pixels
    padded_width = square_size * no_squares_w
    padded_height = square_size * no_squares_h
    padded_arr = np.zeros((padded_height, padded_width), dtype=np.uint8)

    for i in range(arr.shape[0]):
        print("Row: ", i, "; columns: ", (arr.shape[1]), "; arr.shape: ", arr.shape, "; padd_arr.shape: ", padded_arr.shape)
        for j in range(arr.shape[1]):
            padded_arr[i, j] = arr[i, j]

    result = np.zeros((no_squares_w, no_squares_h, square_size, square_size))

    for i in range(no_squares_h):
        for j in range(no_squares_w):
            result[j, i] = padded_arr[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size]

    print("Result.shape [split_into_squares]: ", result.shape)

    return result

def reconstruct_image(arr):
    """
    Reconstructs an image from a structured array.
    Supports multi-channel images with shape (C, num_cols, num_rows, square_w, square_h)
    or grayscale images with shape (num_cols, num_rows, square_w, square_h).

    Parameters:
        arr (numpy.ndarray): Input array with shape (C, num_cols, num_rows, square_w, square_h) for RGB,
                             or (num_cols, num_rows, square_w, square_h) for grayscale.

    Returns:
        numpy.ndarray: Reconstructed image with shape (C, full_width, full_height) for RGB,
                       or (full_width, full_height) for grayscale.
    """
    if arr.ndim == 5:  # Multi-channel (RGB or similar)
        C, num_cols, num_rows, square_w, square_h = arr.shape
        return arr.transpose(0, 2, 1, 4, 3).reshape(C, num_rows * square_h, num_cols * square_w)
    elif arr.ndim == 4:  # Grayscale
        num_cols, num_rows, square_w, square_h = arr.shape
        return arr.transpose(1, 0, 3, 2).reshape(num_rows * square_h, num_cols * square_w)
    else:
        raise ValueError("Input array must have 4 (grayscale) or 5 (multi-channel) dimensions.")


def convolve_anim(img, kernel, partition=10, pad_val=0, logging=False):
    #partition_result = np.zeros_like(img)
    print("Img shape: ", np.shape(img))

    channels = []
    partition_result = []

    if img.ndim == 2:
        partition_result = split_into_squares(img, partition)
    elif img.ndim == 3:
        print("img.shape[2]: ", img.shape[2])
        for i in range(img.shape[2]):
            img_channel = img[:, :, i]
            res = split_into_squares(img_channel, partition)
            print("Split_into_squares result shape: ", np.shape(res))

            channels.append(np.expand_dims(res, axis=0))

        partition_result = np.concatenate(channels, axis=0)

        print("Partition_result shape: ", np.shape(partition_result))

        reconstructed = reconstruct_image(partition_result)

        print("Reconstructed image shape: ", np.shape(reconstructed))

        print("Channels.shape: ", np.shape(channels))

        _, ax5 = plt.subplots(1, 3, figsize=(10, 5))
        ax5[0].imshow(channels[0][0][0][0], cmap='Reds')
        ax5[1].imshow(channels[1][0][0][0], cmap='Greens')
        ax5[2].imshow(channels[2][0][0][0], cmap='Blues')

        plt.show()


        reconstructed = reconstructed.astype(np.uint8)

        plt.imshow(np.transpose(reconstructed, (1, 2, 0)))  # Convert (C, H, W) -> (H, W, C)
        plt.axis("off")  # Hide axes
        plt.show()

    else:
        print('Wrong img type [convolve_anim()]!')
        return partition_result

    return partition_result

def convolve2d(img, kernel, pad_val=0, logging=False):
    if logging: print("Image size [convolution]: ", np.shape(img))

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
                    if ch == 2 and j % 10 == 0: result_animation.append(result.copy())

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
