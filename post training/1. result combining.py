import numpy as np
import os

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray')

data = np.load('data_testing.npy')      # change
data_len = len(data)
print(data_len)

########################################################################################################################
# stretch the file names

filename = 'testing'      # change to your predicting data name
file_name = os.listdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/yolo predicting data/{}/labels'.format(filename))
file_name.sort(key=lambda x:int(x[:-4]))

length = len(file_name)
# print(length)

for i in range(length):
    basename = file_name[i]
    file_name[i] = os.path.splitext(basename)[0]

file_name = np.array(file_name, dtype=int)
print(file_name.shape)
print(file_name)
########################################################################################################################

os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/yolo predicting data/{}/labels'.format(filename))      # change

labels = np.zeros((data_len, 5), float)

for i in range(length):
    # for line in open("{}.txt".format(file_name[i])):
    #     labels[i] = np.loadtxt(labels, np.fromstring(line, dtype=float, sep=' '))
    temp = np.loadtxt("{}.txt".format(file_name[i]), dtype=float, delimiter=' ')

    for j in range(5):
        labels[file_name[i]-1][j] = temp[j]

print(labels)

########################################################################################################################
# save txt
os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/predicting result in txt')
np.savetxt('{}.txt'.format(filename), labels, fmt='%.6f', delimiter='\t')       # change
