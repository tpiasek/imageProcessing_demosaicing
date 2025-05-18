import skimage as sk
from src import cfa_simulation as cfa_sim
import matplotlib.pyplot as plt
import numpy as np
from src import convolution as conv
import matplotlib.animation as animation

img_m = sk.io.imread("img/test_micro.jpg")  # Load image
img_m = cfa_sim.simulate_cfa_3d(img_m)

kernel = np.array([[0.25, 0.5, 0.25],
                       [0.5, 1, 0.5],
                       [0.25, 0.5, 0.25]], dtype='float64')

print("Process starts...")

img_res = conv.convolve_anim(img_m, kernel, pad_val=0, logging=True)

print("Process finished")