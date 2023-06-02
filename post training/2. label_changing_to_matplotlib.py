import numpy as np
import os
import math

os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/predicting result in txt')

filename = 'testing'           # change the filename to the name you want

temp = np.loadtxt('{}.txt'.format(filename), delimiter='\t', dtype=float)      # change
print(temp.shape)


labels = np.zeros((len(temp), 4), int)

for i in range(len(temp)):
    labels[i][2] = round(temp[i][3] * 24)
    labels[i][3] = round(temp[i][4] * 32)
    labels[i][0] = round((temp[i][1] * 24) - (labels[i][2]/2))
    labels[i][1] = round((temp[i][2] * 32) - (labels[i][3]/2))


print(labels)

labels = np.array(labels, dtype=int)


os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/converted result/ndarray')
np.save('{}.npy'.format(filename), labels)

os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/converted result/txt')
np.savetxt('{}.txt'.format(filename), labels, fmt='%.i', delimiter='\t')