import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data in ndarray')

data = np.load('data_1003_1.npy')

print(len(data))

