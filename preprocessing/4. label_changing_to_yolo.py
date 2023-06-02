import numpy as np
import os

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/labels/label in ndarray')         # change the direction to your own direction
label = np.load('label_Joe_aug.npy')            # change filename to which you wanna load
print(label.shape)

label_yolo = np.zeros((len(label), 5), float)

for i in range(len(label)):
    xmin = label[i][0]
    ymin = label[i][1]
    xmax = label[i][0] + label[i][2]
    ymax = label[i][1] + label[i][3]

    label_yolo[i][0] = 0                         # class
    label_yolo[i][1] = ((xmin + xmax)/2)/24      # x_center
    label_yolo[i][2] = ((ymin + ymax)/2)/32      # y_center
    label_yolo[i][3] = (xmax - xmin)/24          # width
    label_yolo[i][4] = (ymax - ymin)/32          # height


print(label_yolo.shape)


os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/yolo label/yolo label in txt')
np.savetxt('training_label_yolo.txt', label_yolo, fmt='%f', delimiter='\t')          # change saving name as you want

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/yolo label/yolo label in ndarray')
np.save('training_label_yolo.npy', label_yolo)          # change saving name as you want