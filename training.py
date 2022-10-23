import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

########################################################################################################################
#data processing
########################################################################################################################
#
# data1 = np.load('data_Joe.npy')
# data2 = np.load('data_0921_1.npy')
# data3 = np.load('data_0922_1.npy')
# data4 = np.load('data_0922_2.npy')
#
# label1 = np.load('label_Joe.npy')
# label2 = np.load('label_0921_1.npy')
# label3 = np.load('label_0922_1.npy')
# label4 = np.load('label_0922_2.npy')
#
#
train_data = np.load('data_Joe_aug.npy')
train_label_temp = np.load('label_Joe_aug.npy')
#
# train_label_Joe_temp = np.append(label1, label2, axis=0)
# train_label_Joe_temp = np.append(train_label_Joe_temp, label3, axis=0)
# train_label_Joe_temp = np.append(train_label_Joe_temp, label4, axis=0)
# # train_label_Joe_temp = train_label_Joe_temp.reshape(-1, 4)
# print('train data joe shape: ', train_data_Joe.shape)
# print('train label joe temp shape: ', train_label_Joe_temp.shape)
# # print(train_label_Joe[0])
#
train_label = np.empty((len(train_data), ), float)
err = 0
# print(train_label_Joe_temp[1][1])
#
for i in range(len(train_label_temp)):
    # print(train_label_temp[i][0])
    train_label[i] = (train_label_temp[i][1] - 1)*24 + train_label_temp[i][0]
    if train_label[i] < 0:
        train_label[i] = 0
#
# print('train label: ', train_label)

########################################################################################################################
#setting training sets
########################################################################################################################

#here should be a normalization step

training_set, val_set, training_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

training_set = torch.from_numpy(training_set)
# print(training_set.dtype)
training_label = torch.from_numpy(training_label).type(torch.LongTensor)
# training_label = training_set.to(torch.float32)

val_set = torch.from_numpy(val_set)
val_label = torch.from_numpy(val_label).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(training_set, training_label)
print(train)
val = torch.utils.data.TensorDataset(val_set, val_label)


LR = 0.001
batch_size = 100
num_epochs = 100

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)


########################################################################################################################
#CNN Model
########################################################################################################################

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,32,24)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)  # output_shape=(4,28,20) # output shape 2=(8,30,22)
        self.relu1 = nn.ReLU()  # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # output_shape=(4,14,10) # output shape 2=(4,15,11)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=0)  # output_shape=(8,10,6) # output shape 2=(32,13,9)
        self.relu2 = nn.ReLU()  # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # output_shape=(8,5,3) # output shape 2=(16,7,5)
        # Fully connected 1 ,#input_shape=(32*4*4)
        # self.fc1 = nn.Linear(16 * 7 * 5, 768)   # softmax change size
        self.fc1 = nn.Linear(32 * 6 * 4, 768)
        # self.fc1 = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        # Convolution 1
        # print('x1 shape: ', x.shape)
        out = self.cnn1(x)
        # print('x2 shape: ', out.shape)
        out = self.relu1(out)     # cancelled
        # print("x2', shape: ", out.shape)
        # Max pool 1
        out = self.maxpool1(out)
        # print('x3 shape: ', out.shape)
        # Convolution 2
        out = self.cnn2(out)
        # print('x4 shape: ', out.shape)
        out = self.relu2(out)     # cancelled
        # print("x4' shape: ", out.shape)
        # Max pool 2
        out = self.maxpool2(out)
        # print('x5 shape: ', out.shape)
        out = self.relu2(out)
        # print('x6 shape: ', out.shape)
        out = out.view(out.size(0), -1)
        # print('x7 shape: ', out.shape)
        # Linear function (readout)
        out = self.fc1(out)
        # print('x8 shape: ', out.shape)
        return out

########################################################################################################################
#parameter setting
########################################################################################################################

model = CNN_Model().double()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
input_shape = (-1, 1, 32, 24)

########################################################################################################################
#training
########################################################################################################################

def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader):
    # Training the Model
    #history-like list for store loss & acc value
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            # print('1. ', images.shape)
            # print('2. ', labels.shape)
            # 1.Define variables
            train = Variable(images.view(input_shape))
            # print('train shape: ', train.shape)
            # print('train data type: ', train.dtype)
            labels = Variable(labels)
            # print('labels shape: ', labels.shape)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train)
            # print('output shape: ', outputs.shape)
            # print('labels shape: ', labels.shape)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for images, labels in val_loader:
            # 1.Define variables
            test = Variable(images.view(input_shape))
            # 2.Forward propagation
            outputs = model(test)
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Training_Loss: {:.4f} Training_acc: {:.6f}% Val_Loss: {:.4f} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    return training_loss, training_accuracy, validation_loss, validation_accuracy


training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader)





########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
#setting training sets num2
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


#here should be a normalization step

# training_set = train_data
# training_label = train_label
#
# training_set = torch.from_numpy(training_set)
# training_label = torch.from_numpy(training_label).type(torch.LongTensor)
#
# train = torch.utils.data.TensorDataset(training_set, training_label)
#
# # testing
#
# test_data_Han = np.load('data_Han_aug.npy')
# test_label_Han_temp = np.load('label_Han_aug.npy')
# print('test label han temp shape: ', test_label_Han_temp.shape)
#
# test_label_Han = np.empty((len(test_label_Han_temp), ), float)
#
# for i in range(len(test_label_Han_temp)):
#     test_label_Han[i] = (test_label_Han_temp[i][1] - 1)*24 + test_label_Han_temp[i][0]
#     if test_label_Han[i] < 0:
#         test_label_Han[i] = 0
#
# testing_data = torch.from_numpy(test_data_Han)
# testing_label = torch.from_numpy(test_label_Han).type(torch.LongTensor)  # data type is long
# testing = torch.utils.data.TensorDataset(testing_data, testing_label)
# testing_loader = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)
#
#
# LR = 0.001
# batch_size = 100
# num_epochs = 10
#
# train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
# # val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
#
#
# ########################################################################################################################
# #CNN Model
# ########################################################################################################################
#
# class CNN_Model(nn.Module):
#     def __init__(self):
#         super(CNN_Model, self).__init__()
#         # Convolution 1 , input_shape=(1,32,24)
#         self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)  # output_shape=(4,28,20) # output shape 2=(4,30,22)
#         self.relu1 = nn.ReLU()  # activation
#         # Max pool 1
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # output_shape=(4,14,10) # output shape 2=(4,15,11)
#         # Convolution 2
#         self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=0)  # output_shape=(8,10,6) # output shape 2=(16,13,9)
#         self.relu2 = nn.ReLU()  # activation
#         # Max pool 2
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # output_shape=(8,5,3) # output shape 2=(16,7,5)
#         # Fully connected 1 ,#input_shape=(32*4*4)
#         # self.fc1 = nn.Linear(16 * 7 * 5, 768)   # softmax change size
#         self.fc1 = nn.Linear(768, 768)
#         # self.fc1 = nn.Linear(32 * 6 * 6, 10)
#
#     def forward(self, x):
#         # Convolution 1
#         # print('x1 shape: ', x.shape)
#         out = self.cnn1(x)
#         # print('x2 shape: ', out.shape)
#         out = self.relu1(out)     # cancelled
#         # print("x2', shape: ", out.shape)
#         # Max pool 1
#         out = self.maxpool1(out)
#         # print('x3 shape: ', out.shape)
#         # Convolution 2
#         out = self.cnn2(out)
#         # print('x4 shape: ', out.shape)
#         out = self.relu2(out)     # cancelled
#         # print("x4' shape: ", out.shape)
#         # Max pool 2
#         out = self.maxpool2(out)
#         # print('x5 shape: ', out.shape)
#         out = self.relu2(out)
#         # print('x6 shape: ', out.shape)
#         out = out.view(out.size(0), -1)
#         # print('x7 shape: ', out.shape)
#         # Linear function (readout)
#         out = self.fc1(out)
#         # print('x8 shape: ', out.shape)
#         return out
#
# ########################################################################################################################
# #parameter setting
# ########################################################################################################################
#
# model = CNN_Model().double()
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
# loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
# input_shape = (-1, 1, 32, 24)
#
# ########################################################################################################################
# #training
# ########################################################################################################################
#
# def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, testing_loader):
#     # Training the Model
#     #history-like list for store loss & acc value
#     training_loss = []
#     training_accuracy = []
#     testing_loss = []
#     testing_accuracy = []
#     for epoch in range(num_epochs):
#         #training model & store loss & acc / epoch
#         correct_train = 0
#         total_train = 0
#         for images, labels in train_loader:
#             # print('1. ', images.shape)
#             # print('2. ', labels.shape)
#             # 1.Define variables
#             train = Variable(images.view(input_shape))
#             # print('train shape: ', train.shape)
#             # print('train data type: ', train.dtype)
#             labels = Variable(labels)
#             # print('labels shape: ', labels.shape)
#             # 2.Clear gradients
#             optimizer.zero_grad()
#             # 3.Forward propagation
#             outputs = model(train)
#             # print('output shape: ', outputs.shape)
#             # print('labels shape: ', labels.shape)
#             # 4.Calculate softmax and cross entropy loss
#             train_loss = loss_func(outputs, labels)
#             # 5.Calculate gradients
#             train_loss.backward()
#             # 6.Update parameters
#             optimizer.step()
#             # 7.Get predictions from the maximum value
#             predicted = torch.max(outputs.data, 1)[1]
#             # 8.Total number of labels
#             total_train += len(labels)
#             # 9.Total correct predictions
#             correct_train += (predicted == labels).float().sum()
#         #10.store val_acc / epoch
#         train_accuracy = 100 * correct_train / float(total_train)
#         training_accuracy.append(train_accuracy)
#         # 11.store loss / epoch
#         training_loss.append(train_loss.data)
#
#         #evaluate model & store loss & acc / epoch
#         correct_test = 0
#         total_test = 0
#         for test, labels in (testing_loader):
#             # 1.Define variables
#             test = Variable(images.view(input_shape))
#             labels = Variable(labels)
#             # 2.Forward propagation
#             outputs = model(test)
#             # 3.Calculate softmax and cross entropy loss
#             test_loss = loss_func(outputs, labels)
#             # 4.Get predictions from the maximum value
#             predicted = torch.max(outputs.data, 1)[1]
#             # 5.Total number of labels
#             total_test += len(labels)
#             # 6.Total correct predictions
#             correct_test += (predicted == labels).float().sum()
#         #6.store val_acc / epoch
#         test_accuracy = 100 * correct_test / float(total_test)
#         testing_accuracy.append(test_accuracy)
#         # 11.store val_loss / epoch
#         testing_loss.append(test_loss.data)
#         print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% testing_Loss: {} testing_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, testing_loss.data, testing_accuracy))
#     return training_loss, training_accuracy, testing_loss, testing_accuracy
#
#
# training_loss, training_accuracy, testing_loss, testing_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, testing_loader)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################




plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
plt.title('Training & Testing loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
plt.title('Training & Validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




# plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
# plt.plot(range(num_epochs), testing_loss, 'g-', label='Testing_loss')
# # plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
# # plt.title('Training & Validation loss')
# plt.title('Training & Testing loss')
# plt.xlabel('Number of epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
# plt.plot(range(num_epochs), testing_accuracy, 'g-', label='Testing_accuracy')
# # plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
# # plt.title('Training & Validation accuracy')
# plt.title('Training & Testing accuracy')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()





########################################################################################################################
#testing
########################################################################################################################
# batch = 1
test_data_Han = np.load('data_Han_aug.npy')
test_label_Han_temp = np.load('label_Han_aug.npy')
print('test label han temp shape: ', test_label_Han_temp.shape)
# test_label_Han_temp = test_label_Han_temp.reshape(-1, 4)
# print(len(test_label_Han_temp))

test_label_Han = np.empty((len(test_label_Han_temp), ), float)

for i in range(len(test_label_Han_temp)):
    test_label_Han[i] = (test_label_Han_temp[i][1] - 1)*24 + test_label_Han_temp[i][0]
    if test_label_Han[i] < 0:
        test_label_Han[i] = 0

testing_data = torch.from_numpy(test_data_Han)
testing_label = torch.from_numpy(test_label_Han).type(torch.LongTensor)  # data type is long
print('testing data shape: ', testing_data.shape)
print('testing label shape: ', testing_label.shape)
testing = torch.utils.data.TensorDataset(testing_data, testing_label)
testing_loader = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)



counter = 0
err_sum = 0
predictlist = []
labellist = []
testing_result = np.empty((100, 1, 32, 24), float)
labeling_result = np.empty((100), float)
predicted_result = np.empty((100), float)

for images, labels in (testing_loader):
    test_test = Variable(images.view(input_shape))
    labels = Variable(labels)

    outputs = model(test_test)
    predicted = torch.max(outputs.data, 1)[1]


    test_test = test_test.numpy()  # new testing
    predicted = predicted.numpy()
    labels = labels.numpy()


    if (counter <= 9):
        print('label: ', labels[:10])
        print('prediction: ', predicted[:10])
        # c = 0
        for i in range(10):
            testing_result[counter*10 + i] = test_test[i]
            predicted_result[counter*10 + i] = predicted[i]
            labeling_result[counter*10 + i] = labels[i]

            predictlist.append(predicted[i])     # old
            labellist.append(labels[i])

        err_counter = 0

        for i in range(10):
            if labels[i] != predicted[i]:
                err_counter += 1
                err_sum += 1

        print('numbers of error: ', err_counter)

    counter += 1



########################################################################################################################
#confusion matrix
########################################################################################################################

testing_result = testing_result.squeeze(1)
print('testing result shape: ', testing_result.shape)


print('total error: ', err_sum)
print('confusion matrix: ')
print(confusion_matrix(labellist, predictlist))

print('testing accuracy: ', (100 - err_sum) / 100 * 100, '%')
########################################################################################################################
# visualization
########################################################################################################################

# plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
# plt.plot(range(num_epochs), testing_loss, 'g-', label='testing_loss')
# # plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
# # plt.title('Training & Validation loss')
# plt.title('Training & Testing loss')
# plt.xlabel('Number of epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
# plt.plot(range(num_epochs), testing_accuracy, 'g-', label='Validation_accuracy')
# # plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
# # plt.title('Training & Validation accuracy')
# plt.title('Training & Testing accuracy')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()



for i in range(50):
    label_2 = (labeling_result[i] // 24) + 1
    label_1 = labeling_result[i] - ((label_2 - 1)*24)
    predicted_2 = (predicted_result[i] // 24) + 1
    predicted_1 = predicted_result[i] - ((predicted_2 - 1)*24)

    fig, ax = plt.subplots()
    ax.imshow(testing_result[i], vmin=22, vmax=32.5, cmap='jet')

    fig, ax = plt.subplots()
    ax.imshow(testing_result[i], vmin=22, vmax=32.5, cmap='jet')
    rect1 = patches.Rectangle((label_1, label_2), 5, 5, lw=2, ec='r', fc='none')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((predicted_1, predicted_2), 5, 5, lw=2, ec='y', fc='none')
    ax.add_patch(rect2)



plt.show()