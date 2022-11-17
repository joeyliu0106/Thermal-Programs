import numpy as np
import os

os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data base')


label = np.loadtxt('label_Han_aug.txt', dtype=int)
print(label.shape)

label_yolo = np.zeros((len(label), 5), float)

for i in range(len(label)):
    xmin = label[i][0]
    ymin = label[i][1]
    xmax = label[i][0] + 5
    ymax = label[i][1] + 5

    label_yolo[i][0] = 0                         # class
    label_yolo[i][1] = ((xmin + xmax)/2)/24      # x_center
    label_yolo[i][2] = ((ymin + ymax)/2)/32      # y_center
    label_yolo[i][3] = (xmax - xmin)/24          # width
    label_yolo[i][4] = (ymax - ymin)/32          # height


print(label_yolo.shape)


os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data base/yolo label')

np.savetxt('testing_label_yolo.txt', label_yolo, fmt='%f', delimiter='\t')