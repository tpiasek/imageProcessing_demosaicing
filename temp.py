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

print("Convolution starts...")
img_res, img_anim = conv.convolve2d(img_m, kernel, logging=True)
print("Convolution done.")
img_res = conv.normalize_image(img_res)
for i in range(len(img_anim)):
    img_anim[i] = conv.normalize_image(img_anim[i])

fig = plt.figure(figsize=(8,8))
ax1 = plt.axes(xlim=(0, img_res.shape[1]), ylim=(0, img_res.shape[0]))
img_plot = plt.imshow(img_anim[0], interpolation='none')

img_anim = np.array(img_anim)

print(img_anim.shape)
print("img_anim length: ", len(img_anim))

def animate(frame):
    img_plot.set_array(np.flipud(img_anim[frame]))
    return [img_plot]

anim = animation.FuncAnimation(
    fig,
    animate,
    frames = int(len(img_anim)),
    interval = 1,
)

writergif = animation.PillowWriter()
anim.save('img/test_anim.gif', writer=writergif)

_, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].imshow(img_m)
ax[1].imshow(img_res)

print(img_m)

# sk.io.imsave("img/test_mosaiced.jpg", np.ndarray.astype(img_m, np.uint8))

plt.show()
