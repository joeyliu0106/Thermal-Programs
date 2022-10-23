import numpy as np
import os

os.chdir('C:/Users/Joe/Desktop/thermal_cnn/program/data base')


label = np.loadtxt('label_Joe.txt', dtype=int)
length, _ = label.shape

for i in range(length):
    label[i][0] += 2
    label[i][1] += 2

np.savetxt('label_Joe.txt', label, fmt='%i', delimiter='\t')