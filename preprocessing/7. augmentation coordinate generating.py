import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image

###########   data loading   ###########
filename = ''
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/{}'.format(filename))
data = np.load('data_Joe_aug.npy')              # change the name to the file you wanna load

filename = ''
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/labels/label in ndarray/{}'.format(filename))
label = np.load('label_Joe_aug.npy')            # change the name to the file you wanna load
num, length, width = data.shape


###########   random coordinate recording   ###########
aug_loc = np.zeros((num, 2), dtype=int)


###########   random coordinate generating   ###########
for n in range(num):
    x_rand, y_rand = np.random.randint(low=0, high=[[19], [27]], size=(2, 1), dtype=int)

    label_loc_num = round((label[n, 2] + 1) * (label[n, 3] + 1))
    label_loc = np.zeros((label_loc_num), dtype=int)
    random_loc = np.zeros((36), dtype=int)

    # label location calc
    counter = 0

    for l in range(round(label[n, 3] + 1)):
        for w in range(round(label[n, 2] + 1)):
            label_loc[counter] = round(((label[n, 1] + l) * 24 + (label[n, 0] + w)))
            counter += 1

    # random location calc
    counter = 0

    for l in range(6):
        for w in range(6):
            random_loc[counter] = (y_rand + l) * 24 + (x_rand + w)
            counter += 1

    # overlap checking
    err_count = 0

    for rand in range(36):
        for lab in range(label_loc_num):
            if random_loc[rand] == label_loc[lab]:
                err_count += 1

    if err_count != 0:
        n = n - 1
    elif err_count == 0:
        aug_loc[n, 0] = x_rand
        aug_loc[n, 1] = y_rand


###########   file saving   ###########
filename = 'training_augmentation'              # change the name as you want
path = 'C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/augmenting coordinate/{}'.format(filename)
if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)

np.save('aug_coordinate.npy', aug_loc)
print(aug_loc)
print('done')