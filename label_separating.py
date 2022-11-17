import numpy as np
import os

os.chdir('C:/Users/Joe/Desktop/thermal_cnn/data base/yolo label')


label_temp = np.loadtxt('training_label_yolo.txt', dtype=np.float32)
label_temp = label_temp.reshape(len(label_temp), 1, 5)
print(label_temp.shape)


os.chdir('C:/Users/Joe/Desktop/yolov5/datasets/thermal detection/labels/train')


for i in range(len(label_temp)):
    np.savetxt('{}.txt'.format(i + 1), label_temp[i], fmt='%f', delimiter=' ')


