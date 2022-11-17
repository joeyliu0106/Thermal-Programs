import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data in ndarray')

data = np.load('data_1027_3.npy')

os.mkdir('C:/Users/Joe/Desktop/thermal_cnn/data base/data in pic/1027_3')
os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data base/data in pic/1027_3')

counter = 0

for i in range(len(data)):
    plt.imsave('temp.jpg', data[i], vmin=24, vmax=32.5, cmap='jet')

    image = cv2.imread('temp.jpg')
    image = cv2.resize(image, (360, 480), interpolation=cv2.INTER_AREA)

    cv2.imwrite('{}.jpg'.format(i+1), image)
    counter += 1

    if counter == len(data):
        os.remove('temp.jpg')

print(len(data))

