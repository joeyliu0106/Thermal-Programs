import numpy as np
import time
import os


os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data base/not processed data')

########################################################################################################################
#array setting
########################################################################################################################
data_temp = np.empty((0, 0), float)
########################################################################################################################
#data loading
########################################################################################################################
counter = 0

# for line in open("ThermalData_20221003_1.txt"):                                                     # parameter changing
#     if len(line) > 300:
#         if counter <= 199:
#             data_1003_1 = np.append(data_1003_1, np.fromstring(line, dtype=float, sep=' '))
#         elif counter >= 1000 and counter <= 1099:
#             data_1003_1 = np.append(data_1003_1, np.fromstring(line, dtype=float, sep=' '))
#
#         counter += 1


# normal data reading
for line in open("ThermalData_20221027_3.txt"):
    if len(line) > 300:
        data_temp = np.append(data_temp, np.fromstring(line, dtype=float, sep=' '))



# normal label reading
# for line in open("label_Joe_Move.txt"):
#     label_Joe_Move = np.append(label_Joe_Move, np.fromstring(line, dtype=float, sep='\t'))

########################################################################################################################
#data processing(reshape, rotate)
########################################################################################################################
#labels processing

# label_Joe = label_Joe.reshape(-1, 4)
# print(label_Joe.shape)
# print(label_Joe_ud.shape)
# print(label_Joe_lr.shape)
# print(label_Joe[0])


#data processing
data_temp = data_temp.reshape(-1, 24, 32)
data_temp = np.rot90(data_temp, k=-1, axes=(1, 2))
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


os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data in ndarray')

np.save('data_1027_3', data_temp)



# print('training data set size: ', data_Joe_aug.shape)
# print('training label set size: ', label_Joe_aug.shape)