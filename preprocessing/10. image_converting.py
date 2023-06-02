import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image

filename = '3_dim_err_data_fixed_scale_training'          # change
path = 'C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/{}'.format(filename)

os.chdir(path)

err_map = np.load('1st order err map.npy')
err_map_sec = np.load('2nd order err map.npy')
err_map_total = np.load('3 dim err map.npy')

num, _, _ = err_map.shape

############   converting to RGB   #############
filename = '3_dim_err_data_fixed_scale_training'            # change
path = 'C:/Users/nina/Desktop/thermal cnn/data base/data/data in pic/{}'.format(filename)
if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)

for i in range(num):
    image = Image.fromarray(err_map_total[i].astype(np.uint8), 'RGB')
    image.save('{}.jpg'.format(i+1))

############   image checking   ###########

plt.figure()
plt.imshow(err_map[400], vmin=0, vmax=20, cmap='jet')

plt.figure()
plt.imshow(err_map_sec[400], vmin=0, vmax=20, cmap='jet')

plt.figure()
plt.imshow(image)

plt.show()