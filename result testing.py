import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

data_0921_1 = np.load('data_0921_1.npy')
data_0922_1 = np.load('data_0922_1.npy')
data_0922_2 = np.load('data_0922_2.npy')
# data_Han_ud = np.load('data_Han_ud.npy')

########################################################################################################################
#labeling with showing pics
########################################################################################################################

fig, ax = plt.subplots()

ax.imshow(data_0921_1[0], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
# rect1 = patches.Rectangle((11, 7), 5, 5, lw=1, ec='r', fc='none')
# ax.add_patch(rect1)
# rect2 = patches.Rectangle((11, 4), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect2)


fig, ax = plt.subplots()

ax.imshow(data_0921_1[255], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
# rect1 = patches.Rectangle((12, 7), 5, 5, lw=1, ec='r', fc='none')
# ax.add_patch(rect1)
# rect2 = patches.Rectangle((11, 4), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect2)


fig, ax = plt.subplots()

ax.imshow(data_0922_1[0], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
rect1 = patches.Rectangle((10, 4), 5, 5, lw=1, ec='r', fc='none')
ax.add_patch(rect1)
# rect2 = patches.Rectangle((9, 5), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect2)
# rect3 = patches.Rectangle((10, 2), 5, 5, lw=1, ec='white', fc='none')
# ax.add_patch(rect3)

plt.show()


# fig, ax = plt.subplots()
#
# ax.imshow(data_0922_2[0], vmin=25, vmax=32.5, cmap='jet')  # haven't done yet
# rect1 = patches.Rectangle((11, 7), 5, 5, lw=1, ec='r', fc='none')
# ax.add_patch(rect1)
# rect2 = patches.Rectangle((5, 3), 5, 5, lw=1, ec='y', fc='none')
# ax.add_patch(rect2)
#
# plt.show()