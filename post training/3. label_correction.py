import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2
import numpy as np
from PIL import Image


#########   data loading   ###########
filename = 'testing'         # change
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray/{}'.format(filename))
data = np.load('median.npy')
num, length, width = data.shape

os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/converted result/ndarray')
label = np.load('{}.npy'.format(filename))         # change

print(label.shape)


########   correction implementation   #########
buffer = np.zeros((num, 2), dtype=int)
max_pos = np.zeros((num, 2), dtype=int)
label_new = np.zeros((num, 4), dtype=int)
count = 1

for n in range(num):
    print('**********************************************************************')
    print(label[n])
    print(count)



    kernel = np.zeros((label[n, 2] + 1, label[n, 3] + 1, 2), dtype=float)
    value = np.zeros((label[n, 3] + 1, label[n, 2] + 1), dtype=float)

    # kernel label recording
    for l in range(label[n, 3] + 1):
        for w in range(label[n, 2] + 1):
            if label[n, 1] + l > 31:
                break
            elif label[n, 0] + w > 23:
                break

            value[l, w] = data[n, label[n, 1] + l, label[n, 0] + w]

    print(value)

    # second try
    m = np.argmax(value)
    max_pos[n, 1], max_pos[n, 0] = divmod(m, value.shape[1])

    label_new[n, 0] = label[n, 0]
    label_new[n, 1] = label[n, 1]

    # print(value)
    print(max_pos[n])
    counter = 0

    while (max_pos[n, 0] == 0 or max_pos[n, 0] == label[n, 2] or max_pos[n, 1] == 0 or max_pos[n, 1] == label[n, 3]):

        if max_pos[n, 0] == 0:            # left
            print('left')
            label_new[n, 0] = label_new[n, 0] - 1
            label_new[n, 1] = label_new[n, 1]
            if label_new[n, 0] < 0:
                label_new[n, 0] = 0
                break

        elif max_pos[n, 0] == label[n, 2]:            # right
            print('right')
            label_new[n, 0] = label_new[n, 0] + 1
            label_new[n, 1] = label_new[n, 1]
            if label_new[n, 0] > 23:
                label_new[n, 0] = 23
                break

        elif max_pos[n, 1] == 0:            # up
            print('up')
            label_new[n, 0] = label_new[n, 0]
            label_new[n, 1] = label_new[n, 1] - 1
            if label_new[n, 1] < 0:
                label_new[n, 1] = 0
                break

        elif max_pos[n, 1] == label[n, 3]:            # down
            print('down')
            label_new[n, 0] = label_new[n, 0]
            label_new[n, 1] = label_new[n, 1] + 1
            if label_new[n, 1] > 31:
                label_new[n, 1] = 31
                break

        for l in range(label[n, 3] + 1):
            for w in range(label[n, 2] + 1):
                if label_new[n, 1] + l > 31:
                    break
                elif label_new[n, 0] + w > 23:
                    break

                value[l, w] = data[n, label_new[n, 1] + l, label_new[n, 0] + w]

        print(value)

        m = np.argmax(value)
        max_pos[n, 1], max_pos[n, 0] = divmod(m, value.shape[1])

        print(max_pos[n])

        if counter > 20:
            break

        counter += 1

        # label_new[n, 0] = label[n, 0]
        # label_new[n, 1] = label[n, 1]

    label_new[n, 2] = label[n, 2]
    label_new[n, 3] = label[n, 3]
    print(label_new[n])

    count += 1


print(label_new)


##########   data saving   ###########
path = 'C:/Users/nina/Desktop/thermal cnn/yolo prediction/corrected label/{}'.format(filename)
if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)

np.save('label_corrected.npy', label_new)


##########   image checking   ##########
fig, ax = plt.subplots()

ax.imshow(data[0], vmin=22, vmax=32.5, cmap='jet')
rect = patches.Rectangle((label_new[0, 0], label_new[0, 1]), label_new[0, 2], label_new[0, 3], lw=2, ec='r', fc='none')
ax.add_patch(rect)

plt.show()
