import numpy as np
import os

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/yolo label/yolo label in ndarray')
label_temp = np.load('training_label_yolo.npy')          # change the name to the file you wanna load
label_temp = label_temp.reshape(len(label_temp), 1, 5)
print(label_temp.shape)



filename = 'training_label_yolo'            # change the name to what you prefer
path = 'C:/Users/nina/Desktop/thermal cnn/data base/yolo label/yolo label in ndarray/separated labels/{}'.format(filename)
if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)


for i in range(len(label_temp)):
    np.savetxt('{}.txt'.format(i + 1), label_temp[i], fmt='%f', delimiter=' ')


