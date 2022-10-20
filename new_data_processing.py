import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
# counter = 0
#
# for line in open("ThermalData_20220921_1.txt"):
#     if len(line) > 300:
#         if counter <= 199:
#             data_0921_1 = np.append(data_0921_1, np.fromstring(line, dtype=float, sep=' '))
#         elif counter >= 1000 and counter <= 1099:
#             data_0921_1 = np.append(data_0921_1, np.fromstring(line, dtype=float, sep=' '))
#
#         counter += 1

for line in open("label_Joe_Move.txt"):
    label_Joe_Move = np.append(label_Joe_Move, np.fromstring(line, dtype=float, sep='\t'))

########################################################################################################################
#data processing(reshape, rotate)
########################################################################################################################
#labels processing

label_Joe = label_Joe.reshape(-1, 4)
label_Han = label_Han.reshape(-1, 4)
label_0921_1 = label_0921_1.reshape(-1, 4)
label_0922_1 = label_0922_1.reshape(-1, 4)
label_0922_2 = label_0922_2.reshape(-1, 4)
label_Joe_lr = label_Joe_lr.reshape(-1, 4)
label_Joe_ud = label_Joe_ud.reshape(-1, 4)
label_Han_lr = label_Han_lr.reshape(-1, 4)
label_Han_ud = label_Han_ud.reshape(-1, 4)
label_Joe_Move = label_Joe_Move.reshape(-1, 4)
# print(label_Joe.shape)
# print(label_Joe_ud.shape)
# print(label_Joe_lr.shape)
# print(label_Joe[0])


#data processing
data_Joe = data_Joe.reshape(-1, 24, 32)
data_0921_1 = np.rot90(data_0921_1, k=-1, axes=(1, 2))
# print('data_Joe shape: ', data_Joe.shape)
# print('data_Han shape: ', data_Han.shape)
# print('data_0921_1 shape: ', data_0921_1.shape)
# print('data_0922_1 shape: ', data_0922_1.shape)
# print('data_0922_2 shape: ', data_0922_2.shape)
# print('data_0922_3 shape: ', data_0922_3.shape)
# print('data_0922_4 shape: ', data_0922_4.shape)
# print('data_0922_5 shape: ', data_0922_5.shape)
# print('data_1003_1 shape: ', data_1003_1.shape)
# print('data_Joe_Move shape: ', data_Joe_Move.shape)
# print('data_1013 shape: ', data_1013.shape)

########################################################################################################################
#augmentation
########################################################################################################################
# data_Joe_lr = np.flip(data_Joe, axis=2)
# data_Joe_ud = np.flip(data_Joe, axis=1)
#
# data_Han_lr = np.flip(data_Han, axis=2)
# data_Han_ud = np.flip(data_Han, axis=1)
########################################################################################################################
#augmentation appending
########################################################################################################################
#for hospital
# data_Joe_aug = np.append(data_Joe, data_Han, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_Han_lr, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_Joe_Move, axis=0)
# label_Joe_aug = np.append(label_Joe, label_Han, axis=0)
# label_Joe_aug = np.append(label_Joe_aug, label_Han_lr, axis=0)
# label_Joe_aug = np.append(label_Joe_aug, label_Joe_Move, axis=0)
#
# data_Han_aug = np.append(data_0921_1, data_0922_1, axis=0)
# data_Han_aug = np.append(data_Han_aug, data_0922_2, axis=0)
# # data_Han_aug = np.append(data_Han_aug, data_Han_ud, axis=0)
# label_Han_aug = np.append(label_0921_1, label_0922_1, axis=0)
# label_Han_aug = np.append(label_Han_aug, label_0922_2, axis=0)
# # label_Han_aug = np.append(label_Han_aug, label_Han_ud, axis=0)



# data_Joe_aug = np.append(data_Joe, data_0921_1, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_Han_lr, axis=0)
# # data_Joe_aug = np.append(data_Joe_aug, data_0922_2, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_Joe_Move, axis=0)
# label_Joe_aug = np.append(label_Joe, label_0921_1, axis=0)
# label_Joe_aug = np.append(label_Joe_aug, label_Han_lr, axis=0)
# # label_Joe_aug = np.append(label_Joe_aug, label_0922_2, axis=0)
# label_Joe_aug = np.append(label_Joe_aug, label_Joe_Move, axis=0)
#
# data_Han_aug = np.append(data_Han, data_0922_1, axis=0)
# data_Han_aug = np.append(data_Han_aug, data_0922_2, axis=0)
# # data_Han_aug = np.append(data_Han_aug, data_Han_ud, axis=0)
# label_Han_aug = np.append(label_Han, label_0922_1, axis=0)
# label_Han_aug = np.append(label_Han_aug, label_0922_2, axis=0)
# # label_Han_aug = np.append(label_Han_aug, label_Han_ud, axis=0)

os.chdir('C:/Users/Joe/Desktop/thermal_cnn')


np.save('data_1013', data_1013)



# print('training data set size: ', data_Joe_aug.shape)
# print('training label set size: ', label_Joe_aug.shape)
# print('testing data set size: ', data_Han_aug.shape)
# print('testing label set size: ', label_Han_aug.shape)