import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import figure
import os
import cv2

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray')
data = np.load('data_testing.npy')

data_len = len(data)
print(len(data))


os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/labels/label in ndarray')
testing_labels = np.load('label_testing.npy')


filename = 'testing'
os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/corrected label/{}'.format(filename))      # change
predicting_labels = np.load('label_corrected.npy')



######   picture generating   ######

intersection = [0] * data_len


for i in range(data_len):
    counter = 0
    num_t = 0
    num_p = 0
    testing_num = 0
    predicting_num = 0
    testing_iou = np.zeros((100), dtype=int)
    predicting_iou = np.zeros((100), dtype=int)
    for j in range(int(testing_labels[i][3]+1)):
        for k in range(int(testing_labels[i][2]+1)):
            testing_iou[num_t] = ((testing_labels[i][1]) + j) * 24 + ((testing_labels[i][0]) + k)
            testing_num += 1
            num_t += 1

    for j in range(predicting_labels[i][3] + 1):
        for k in range(predicting_labels[i][2] + 1):
            predicting_iou[num_p] = (predicting_labels[i][1] + j) * 24 + (predicting_labels[i][0] + k)
            predicting_num += 1
            num_p += 1

    for l in range(testing_num):
        for m in range(predicting_num):

            if testing_iou[l] == predicting_iou[m]:
                counter += 1
            else:
                continue


    intersection[i] = counter


print(len(intersection))
intersection = np.array(intersection, dtype=int)


area_t = np.zeros((data_len), int)
area_p = np.zeros((data_len), int)
iou = np.zeros((data_len+3), float)
counter = 0
iou_sum = 0

# iou threshold value
iou_threshold = 0.5


for i in range(data_len):
    area_t[i] = (testing_labels[i][2] + 1) * (testing_labels[i][3] + 1)
    area_p[i] = (predicting_labels[i][2] + 1) * (predicting_labels[i][3] + 1)
    iou[i] = intersection[i] / (area_t[i] + area_p[i] - intersection[i])
    if iou[i] >= iou_threshold:
        counter += 1
    iou_sum += iou[i]



iou[data_len] = counter

# good iou percentage
iou[data_len+1] = counter/data_len

# average iou value
iou[data_len+2] = iou_sum/data_len


print(counter/data_len)             # qualified iou rate
print(iou_sum/data_len)             # iou avg

os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/iou_calc')
np.savetxt('{}_iou_calc.txt'.format(filename), iou, '%.6f')