import numpy as np
import os

###########   direction changing   ###########
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/labels/label in txt')


###########   array setting   ###########
label_temp = np.empty((0, 0), float)


###########   label reading   ###########
for line in open("label_Joe.txt"):
    label_temp = np.append(label_temp, np.fromstring(line, dtype=float, sep='\t'))


###########   label reshaping   ###########
label_temp = label_temp.reshape(-1, 4)


###########   label saving   ###########
os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/labels/label in ndarray')
np.save('label_Joe.npy', label_temp)