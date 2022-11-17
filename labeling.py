import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import time

os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data in ndarray')

data = np.load('data_1003_1.npy')

########################################################################################################################
#labeling with showing pics
########################################################################################################################

fig, ax = plt.subplots()

ax.imshow(data[140], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
rect1 = patches.Rectangle((11, 6), 6, 7, lw=1, ec='r', fc='none')
ax.add_patch(rect1)
# rect2 = patches.Rectangle((11, 7), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect2)
#
# fig, ax = plt.subplots()
#
# ax.imshow(data_Han[0])  # haven't done yet
# rect = patches.Rectangle((11, 4), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect)
#
# fig, ax = plt.subplots()
#
# ax.imshow(data_Han_ud[0])  # haven't done yet
# rect = patches.Rectangle((11, 21), 5, 5, lw=1, ec='r', fc='none')
# ax.add_patch(rect)
#
plt.show()