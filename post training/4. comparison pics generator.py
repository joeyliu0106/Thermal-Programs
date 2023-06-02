import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2

########################################################################################################################
# data loading

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/data/data in ndarray')      # change
data = np.load('data_testing.npy')      # change
print(len(data))

os.chdir('C:/Users/nina/Desktop/thermal cnn/data base/labels/label in ndarray')      # change
testing_labels = np.load('label_testing.npy')      # change


filename = 'testing'
os.chdir('C:/Users/nina/Desktop/thermal cnn/yolo prediction/corrected label/{}'.format(filename))      # change
predicting_labels = np.load('label_corrected.npy')      # change
testing_labels = np.array(testing_labels, dtype=int)

print(testing_labels)
print(predicting_labels)


print(testing_labels.shape)
print(predicting_labels.shape)



########################################################################################################################
# picture generating

path = 'C:/Users/nina/Desktop/thermal cnn/yolo prediction/pics/{}'.format(filename)
if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)



for i in range(700, 900):

    fig, ax = plt.subplots()

    ax.imshow(data[i], vmin=22, vmax=32.5, cmap='jet')  # haven't done yet
    rect1 = patches.Rectangle((testing_labels[i][0], testing_labels[i][1]), testing_labels[i][2], testing_labels[i][3], lw=2, ec='r', fc='none')
    ax.add_patch(rect1)
    rect1 = patches.Rectangle((predicting_labels[i][0], predicting_labels[i][1]), predicting_labels[i][2], predicting_labels[i][3], lw=2, ec='y', fc='none')
    ax.add_patch(rect1)


    plt.savefig('temp.jpg')
    img = cv2.imread('temp.jpg')

    cv2.imwrite('{}.jpg'.format(i+1), img)

