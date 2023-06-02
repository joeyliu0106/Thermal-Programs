import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image

############   data loading   ############
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray')

data = np.load('data_Han_aug.npy')           # change


############   saving direction   ###########
filename = 'testing'            # change the name as you want
path = 'C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/{}'.format(filename)
if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)


############   parameters   ############

num, l, w = data.shape
median = np.zeros((num, l, w), dtype=float)
data_pad = np.zeros((num, l+2, w+2), dtype=float)
number = np.zeros((9), dtype=float)


############   padding   ###########
for i in range(num):
    data_pad[i] = np.pad(data[i], pad_width=(1, 1), mode='edge')


############   median filter   ###########
for n in range(num):
    for i in range(l):
        for j in range(w):
            for x in range(3):
                for y in range(3):
                    number[x*3 + y] = data_pad[n, i-1+x, j-1+y]

            median[n, i, j] = np.median(number)

np.save('median.npy', median)

############   gamma correction   ###########
gamma = np.zeros((num, l, w), dtype=float)
l_bound = np.zeros((num), dtype=float)
h_bound = np.zeros((num), dtype=float)

for n in range(num):
    for i in range(l):
        for j in range(w):
            gamma[n, i, j] = np.power(median[n, i, j] / 36, 1/2)

            ###########   for gamma fixed scale normalization   ###########
            l_bound[n] = np.power(21 / 36, 1/2)
            h_bound[n] = np.power(36 / 36, 1/2)


np.save('gamma_before.npy', gamma)
# print(gamma)

############   fixed scale normalization   ###########
for n in range(num):
    for i in range(l):
        for j in range(w):
            gamma[n, i, j] = int((gamma[n, i, j] - l_bound[n]) / (h_bound[n] - l_bound[n]) * 255)

            if gamma[n, i, j] <= 0:
                gamma[n, i, j] = 0
            elif gamma[n, i, j] >= 255:
                gamma[n, i, j] = 255

norm = gamma




# ###################   if augmentation needed   ###################            # only using when you have to augment the data
# filename = 'training_augmentation'              # change
# os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/augmenting coordinate/{}'.format(filename))
# aug = np.load('augment.npy')
# gamma = np.concatenate((gamma, aug), axis=0)
# num, l, w = gamma.shape
#
#
#
#
# ###################   change direction   #####################
# filename = '3_dim_err_data_fixed_scale_training'            # change
# os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/{}'.format(filename))




# gamma saving
print(gamma.shape)
np.save('gamma.npy', gamma)





###################   1st error map   ###################
data_pad = np.zeros((num, l+2, w+2), dtype=float)
err_map = np.zeros((num, l, w), dtype=float)
err_map_sec = np.zeros((num, l, w), dtype=float)
err_map_pad = np.zeros((num, l+2, w+2), dtype=float)
err_map_total = np.zeros((num, l, w, 3), dtype=float)

for i in range(num):
    data_pad[i] = np.pad(gamma[i], pad_width=(1, 1), mode='edge')

for n in range(num):
    for i in range(l):
        for j in range(w):
            for k in range(3):
                for c in range(3):
                    err_map[n, i, j] += abs(data_pad[n, i + 1, j + 1] - data_pad[n, i + k, j + c])
                    # err_map[n, i, j] += data_pad[n, i + 1, j + 1] - data_pad[n, i + k, j + c]

            err_map[n, i, j] = err_map[n, i, j] / 8

# print('1 order err_map shape: ', err_map.shape)
print('1st order error map saving...')
# saving
np.save('1st order err map.npy', err_map)
print('done')


###################   2nd error map   ###################

for i in range(num):
    err_map_pad[i] = np.pad(err_map[i], pad_width=(1, 1), mode='edge')
# print(err_map_pad.shape)

for n in range(num):
    for i in range(l):
        for j in range(w):
            for k in range(3):
                for c in range(3):
                    err_map_sec[n, i, j] += abs(err_map_pad[n, i + 1, j + 1] - err_map_pad[n, i + k, j + c])

            err_map_sec[n, i, j] = err_map_sec[n, i, j] / 8

# print('2 order err_map shape: ', err_map_sec.shape)
print('2nd order error map saving...')
# saving
np.save('2nd order err map.npy', err_map_sec)
print('done')


###############   error map combining   ################
data = np.concatenate((data, data), axis=0)

for n in range(num):
    for i in range(l):
        for j in range(w):
            err_map_total[n, i, j, 0] = data[n, i, j]
            err_map_total[n, i, j, 1] = err_map[n, i, j]
            err_map_total[n, i, j, 2] = err_map_sec[n, i, j]

print(err_map.shape)
print(err_map_sec.shape)
print('err map total: ', err_map_total)
# saving
print('3 dimension error map saving...')
np.save('3 dim err map.npy', err_map_total)
print('done')


############   image checking   ###########
plt.figure()
plt.imshow(err_map_sec[0], vmin=0, vmax=10, cmap='jet')

plt.figure()
plt.imshow(err_map[0], vmin=0, vmax=10, cmap='jet')

plt.figure()
plt.imshow(norm[0], vmin=0, vmax=255, cmap='jet')

plt.show()