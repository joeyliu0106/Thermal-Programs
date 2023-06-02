import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image


###########   data loading   ###########
filename = '3_dim_err_data_fixed_scale_training'
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data\data in ndarray/{}'.format(filename))
data = np.load('gamma.npy')              # change
num, length, width = data.shape


###########   outlier coordinate loading   ###########
filename = 'training_augmentation'              # change
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/augmenting coordinate/{}'.format(filename))
aug_loc = np.load('aug_coordinate.npy')

for n in range(num):
    x_rand, y_rand = aug_loc[n]

    for l in range(6):
        for w in range(6):
            data[n, y_rand + l, x_rand + w] = 255


print(data[0])


###########   data saving   ###########
np.save('augment.npy', data)