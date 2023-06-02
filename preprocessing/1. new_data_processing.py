import numpy as np
import os


os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/raw data/processed data')            # change to your direction

###########   array setting   ###########
data_temp = np.empty((0, 0), float)


###########   data reading   ###########
counter = 0

# reading data with constraint
# for line in open("ThermalData_20221003_1.txt"):                                                     # parameter changing
#     if len(line) > 300:
#         if counter <= 199:
#             data_1003_1 = np.append(data_1003_1, np.fromstring(line, dtype=float, sep=' '))
#         elif counter >= 1000 and counter <= 1099:
#             data_1003_1 = np.append(data_1003_1, np.fromstring(line, dtype=float, sep=' '))
#
#         counter += 1


# normal data reading
for line in open("ThermalData_Joe.txt"):           # change filename to the one you wanna process with
    if len(line) > 300:
        data_temp = np.append(data_temp, np.fromstring(line, dtype=float, sep=' '))


###########   data processing   ###########
data_temp = data_temp.reshape(-1, 24, 32)
data_temp = np.rot90(data_temp, k=-1, axes=(1, 2))


###########   if data appending needed   ###########        # independ
# # load data for appending
# os.chdir('C:/Users/nina/Desktop/thermal_cnn/data base/data in ndarray')
# data_Joe = np.load('data_Joe.npy')
# data_0921_1 = np.load('data_0921_1.npy')
# data_Joe_Move = np.load('data_Joe_Move.npy')
#
# data_Joe_aug = np.append(data_Joe, data_0921_1, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_Han_lr, axis=0)
# data_Joe_aug = np.append(data_Joe_aug, data_Joe_Move, axis=0)


###########   data saving   ###########
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray')
np.save('data_Joe.npy', data_temp)         # change numpy name as you want