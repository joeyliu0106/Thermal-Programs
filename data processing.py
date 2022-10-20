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
data_Han = np.empty((0, 0), float)
data_Joe_Move = np.empty((0, 0), float)
data_0921_1 = np.empty((0, 0), float)
data_0922_1 = np.empty((0, 0), float)
data_0922_2 = np.empty((0, 0), float)
data_0922_3 = np.empty((0, 0), float)
data_0922_4 = np.empty((0, 0), float)
data_0922_5 = np.empty((0, 0), float)
data_1003_1 = np.empty((0, 0), float)
data_1013 = np.empty((0, 0), float)
label_Joe = np.empty((0, 0), float)
label_Han = np.empty((0, 0), float)
label_Joe_lr = np.empty((0, 0), float)
label_Joe_ud = np.empty((0, 0), float)
label_Han_lr = np.empty((0, 0), float)
label_Han_ud = np.empty((0, 0), float)
label_0921_1 = np.empty((0, 0), float)
label_0922_1 = np.empty((0, 0), float)
label_0922_2 = np.empty((0, 0), float)
label_Joe_Move = np.empty((0, 0), float)


########################################################################################################################
#data loading
########################################################################################################################
for line in open("ThermalData_Han.txt"):
    if len(line) > 300:
        data_Han = np.append(data_Han, np.fromstring(line, dtype=float, sep=' '))

for line in open("ThermalData_Joe.txt"):
    if len(line) > 300:
        data_Joe = np.append(data_Joe, np.fromstring(line, dtype=float, sep=' '))

counter = 1

for line in open("ThermalData_Joe_Move.txt"):
    if len(line) > 300:
        if counter == 20 or counter == 52 or 54 <= counter <= 71 or 105 <= counter <= 131 or counter == 154 or counter == 155 or counter == 179 or counter == 180 or counter == 231 or 255 <= counter <= 311 or counter == 324 or counter == 325 or 337 <= counter <= 339 or counter == 362 or counter == 377 or counter == 378 or counter == 416 or counter == 417 or 431 <= counter <= 441 or counter == 488:
            pass
        else:
            data_Joe_Move = np.append(data_Joe_Move, np.fromstring(line, dtype=float, sep=' '))

        counter += 1

counter = 0

for line in open("ThermalData_20220921_1.txt"):
    if len(line) > 300:
        if counter <= 199:
            data_0921_1 = np.append(data_0921_1, np.fromstring(line, dtype=float, sep=' '))
        elif counter >= 1000 and counter <= 1099:
            data_0921_1 = np.append(data_0921_1, np.fromstring(line, dtype=float, sep=' '))

        counter += 1


counter = 0

for line in open("ThermalData_20220922_1.txt"):
    if len(line) > 300:
        if counter <= 149:
            data_0922_1 = np.append(data_0922_1, np.fromstring(line, dtype=float, sep=' '))

        counter += 1


counter = 0

for line in open("ThermalData_20220922_2.txt"):
    if len(line) > 300:
        if counter <= 199:
            data_0922_2 = np.append(data_0922_2, np.fromstring(line, dtype=float, sep=' '))

        counter += 1


# for line in open("ThermalData_20220922_3.txt"): # broken
#     if len(line) > 300:
#         data_0922_3 = np.append(data_0922_3, np.fromstring(line, dtype=float, sep=' '))

# for line in open("ThermalData_20220922_4.txt"): # broken
#     if len(line) > 300:
#         data_0922_4 = np.append(data_0922_4, np.fromstring(line, dtype=float, sep=' '))

for line in open("ThermalData_20220922_5.txt"): # testing
    if len(line) > 300:
        data_0922_5 = np.append(data_0922_5, np.fromstring(line, dtype=float, sep=' '))

for line in open("ThermalData_20221013.txt"):  # testing
    if len(line) > 300:
        data_1013 = np.append(data_1013, np.fromstring(line, dtype=float, sep=' '))

# for line in open("ThermalData_20221003_1.txt"): # broken
#     if len(line) > 300:
#         data_1003_1 = np.append(data_1003_1, np.fromstring(line, dtype=float, sep=' '))

for line in open("label_Joe.txt"):
    label_Joe = np.append(label_Joe, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_Han.txt"):
    label_Han = np.append(label_Han, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_Joe_lr.txt"):
    label_Joe_lr = np.append(label_Joe_lr, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_Joe_ud.txt"):
    label_Joe_ud = np.append(label_Joe_ud, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_Han_lr.txt"):
    label_Han_lr = np.append(label_Han_lr, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_Han_ud.txt"):
    label_Han_ud = np.append(label_Han_ud, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_20220921_1.txt"):
    label_0921_1 = np.append(label_0921_1, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_20220922_1.txt"):
    label_0922_1 = np.append(label_0922_1, np.fromstring(line, dtype=float, sep='\t'))

for line in open("label_20220922_2.txt"):
    label_0922_2 = np.append(label_0922_2, np.fromstring(line, dtype=float, sep='\t'))

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
data_Han = data_Han.reshape(-1, 24, 32)
data_0921_1 = data_0921_1.reshape(-1, 24, 32)
data_0922_1 = data_0922_1.reshape(-1, 24, 32)
data_0922_2 = data_0922_2.reshape(-1, 24, 32)
data_0922_3 = data_0922_3.reshape(-1, 24, 32)
data_0922_4 = data_0922_4.reshape(-1, 24, 32)
data_0922_5 = data_0922_5.reshape(-1, 24, 32)
data_1003_1 = data_1003_1.reshape(-1, 24, 32)
data_1013 = data_1013.reshape(-1, 24, 32)
data_Joe_Move = data_Joe_Move.reshape(-1, 24, 32)
data_Joe = np.rot90(data_Joe, k=-1, axes=(1, 2))
data_Han = np.rot90(data_Han, k=-1, axes=(1, 2))
data_0921_1 = np.rot90(data_0921_1, k=-1, axes=(1, 2))
data_0922_1 = np.rot90(data_0922_1, k=-1, axes=(1, 2))
data_0922_2 = np.rot90(data_0922_2, k=-1, axes=(1, 2))
data_0922_3 = np.rot90(data_0922_3, k=-1, axes=(1, 2))
data_0922_4 = np.rot90(data_0922_4, k=-1, axes=(1, 2))
data_0922_5 = np.rot90(data_0922_5, k=-1, axes=(1, 2))
data_1003_1 = np.rot90(data_1003_1, k=-1, axes=(1, 2))
data_1013 = np.rot90(data_1013, k=-1, axes=(1, 2))
data_Joe_Move = np.rot90(data_Joe_Move, k=-1, axes=(1, 2))
# print('data_Joe shape: ', data_Joe.shape)
# print('data_Han shape: ', data_Han.shape)
print('data_0921_1 shape: ', data_0921_1.shape)
print('data_0922_1 shape: ', data_0922_1.shape)
print('data_0922_2 shape: ', data_0922_2.shape)
print('data_0922_3 shape: ', data_0922_3.shape)
print('data_0922_4 shape: ', data_0922_4.shape)
print('data_0922_5 shape: ', data_0922_5.shape)
print('data_1003_1 shape: ', data_1003_1.shape)
print('data_Joe_Move shape: ', data_Joe_Move.shape)
print('data_1013 shape: ', data_1013.shape)

########################################################################################################################
#augmentation
########################################################################################################################
data_Joe_lr = np.flip(data_Joe, axis=2)
data_Joe_ud = np.flip(data_Joe, axis=1)

data_Han_lr = np.flip(data_Han, axis=2)
data_Han_ud = np.flip(data_Han, axis=1)

# data_0921_1_lr = np.flip(data_0921_1, axis=2)
# data_0922_1_lr = np.flip(data_0922_1, axis=2)
# data_0922_2_lr = np.flip(data_0922_2, axis=2)

# data_0921_1_ud = np.flip(data_0921_1, axis=1)
# data_0922_1_ud = np.flip(data_0922_1, axis=1)
# data_0922_2_ud = np.flip(data_0922_2, axis=1)
# print('1. ', data_Joe.shape)
# print('2. ', data_Joe_lr.shape)
# print('3. ', data_Joe_ud.shape)
#
# print('4. ', data_Han.shape)
# print('5. ', data_Han_lr.shape)
# print('6. ', data_Han_ud.shape)

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



data_Joe_aug = np.append(data_Joe, data_0921_1, axis=0)
data_Joe_aug = np.append(data_Joe_aug, data_Han_lr, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_0922_2, axis=0)
data_Joe_aug = np.append(data_Joe_aug, data_Joe_Move, axis=0)
label_Joe_aug = np.append(label_Joe, label_0921_1, axis=0)
label_Joe_aug = np.append(label_Joe_aug, label_Han_lr, axis=0)
# label_Joe_aug = np.append(label_Joe_aug, label_0922_2, axis=0)
label_Joe_aug = np.append(label_Joe_aug, label_Joe_Move, axis=0)

data_Han_aug = np.append(data_Han, data_0922_1, axis=0)
data_Han_aug = np.append(data_Han_aug, data_0922_2, axis=0)
# data_Han_aug = np.append(data_Han_aug, data_Han_ud, axis=0)
label_Han_aug = np.append(label_Han, label_0922_1, axis=0)
label_Han_aug = np.append(label_Han_aug, label_0922_2, axis=0)
# label_Han_aug = np.append(label_Han_aug, label_Han_ud, axis=0)

os.chdir('C:/Users/Joe/Desktop/thermal_cnn')



np.save('data_Joe', data_Joe)
np.save('data_Han', data_Han)
np.save('data_Joe_Move', data_Joe_Move)
np.save('label_Joe', label_Joe)
np.save('label_Han', label_Han)
np.save('data_Joe_aug', data_Joe_aug)
np.save('label_Joe_aug', label_Joe_aug)
np.save('data_Han_aug', data_Han_aug)
np.save('label_Han_aug', label_Han_aug)
np.save('data_0921_1', data_0921_1)
np.save('data_0922_1', data_0922_1)
np.save('data_0922_2', data_0922_2)
np.save('label_0921_1', label_0921_1)
np.save('label_0922_1', label_0922_1)
np.save('label_0922_2', label_0922_2)
np.save('data_1003_1', data_1003_1)
np.save('data_Han_lr', data_Han_lr)
np.save('data_Han_ud', data_Han_ud)
np.save('data_1013', data_1013)



print('training data set size: ', data_Joe_aug.shape)
print('training label set size: ', label_Joe_aug.shape)
print('testing data set size: ', data_Han_aug.shape)
print('testing label set size: ', label_Han_aug.shape)
