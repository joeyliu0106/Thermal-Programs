import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import time

os.chdir('C:/Users/Nina/Desktop/thermal_cnn/data base/data in ndarray')

data = np.load('data_Han_aug.npy')

########################################################################################################################
#labeling with showing pics
########################################################################################################################

# fig, ax = plt.subplots()

# ax.imshow(data[700], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
# ax.imshow(data[700])
# rect1 = patches.Rectangle((12, 5), 6, 6, lw=1, ec='r', fc='none')
# ax.add_patch(rect1)
# rect2 = patches.Rectangle((11, 7), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect2)

fig, ax = plt.subplots()

# ax.imshow(data[0])  # haven't done yet
ax.imshow(data[0], vmin=22, vmax=32.5, cmap='jet')
rect = patches.Rectangle((11, 4), 5, 6, lw=1, ec='r', fc='none')
ax.add_patch(rect)
#
# fig, ax = plt.subplots()
#
# ax.imshow(data_Han_ud[0])  # haven't done yet
# rect = patches.Rectangle((11, 21), 5, 5, lw=1, ec='r', fc='none')
# ax.add_patch(rect)
#
# plt.figure()
# plt.imshow(data[0])

# plt.figure()
# plt.imshow(data[0], vmin=22, vmax=32.5, cmap='jet')

plt.show()