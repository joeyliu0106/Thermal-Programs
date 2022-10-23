import numpy as np
import time
import os


os.chdir('C:/Users/Joe/Desktop/thermal_cnn/program/data base')

########################################################################################################################
#array setting
########################################################################################################################
data_Joe = np.empty((0, 0), float)
########################################################################################################################
#data loading
########################################################################################################################
counter = 0

for line in open("ThermalData_20220921_1.txt"):
    if len(line) > 300:
        if counter <= 199:
            data_0921_1 = np.append(data_0921_1, np.fromstring(line, dtype=float, sep=' '))
        elif counter >= 1000 and counter <= 1099:
            data_0921_1 = np.append(data_0921_1, np.fromstring(line, dtype=float, sep=' '))

        counter += 1

for line in open("label_Joe_Move.txt"):
    label_Joe_Move = np.append(label_Joe_Move, np.fromstring(line, dtype=float, sep='\t'))

########################################################################################################################
#data processing(reshape, rotate)
########################################################################################################################
#labels processing

label_Joe = label_Joe.reshape(-1, 4)
# print(label_Joe.shape)
# print(label_Joe_ud.shape)
# print(label_Joe_lr.shape)
# print(label_Joe[0])


#data processing
data_Joe = data_Joe.reshape(-1, 24, 32)
data_0921_1 = np.rot90(data_0921_1, k=-1, axes=(1, 2))
# print('data_Joe shape: ', data_Joe.shape)

########################################################################################################################
#augmentation
########################################################################################################################
# data_Joe_lr = np.flip(data_Joe, axis=2)
# data_Joe_ud = np.flip(data_Joe, axis=1)
########################################################################################################################
#augmentation appending
########################################################################################################################
#for hospital


os.chdir('C:/Users/Joe/Desktop/thermal_cnn')

np.save('data_1013', data_1013)



# print('training data set size: ', data_Joe_aug.shape)
# print('training label set size: ', label_Joe_aug.shape)