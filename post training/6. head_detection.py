import os
import numpy as np

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray')
data = np.load('data_testing.npy')
num, length, width = data.shape

# # not corrected
# os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/converted result/ndarray')         # while using corrected data
# label = np.load('testing.npy')      # change

# corrected
os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/corrected label/testing')         # while using corrected data
label = np.load('label_corrected.npy')      # change


for n in range(num):
    for l in range(length):
        for w in range(width):
            if data[n, l, w] >= 30.5:
                data[n, l, w] = 0


result = np.zeros((num), dtype=int)

for n in range(num):
    value = np.zeros((label[n, 3] + 1, label[n, 2] + 1), dtype=float)

    for l in range(label[n, 3] + 1):
        for w in range(label[n, 2] + 1):
            if label[n, 1] + l > 31:
                break
            elif label[n, 0] + w > 23:
                break

            value[l, w] = data[n, label[n, 1] + l, label[n, 0] + w]

    if np.max(value) == np.max(data[n]):
        result[n] = 1
    else:
        continue

print(result)


##############   ratio calc   ################
counter = 0

for n in range(num):
    if result[n] == 1:
        counter += 1


ratio = (counter / num) * 100

print('ratio of head covering: ', ratio, '%')
