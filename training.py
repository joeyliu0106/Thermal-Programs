import os
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
os.chdir('C:/Users/Joe/Desktop/thermal_cnn')

train_data = np.load('data_Joe_aug.npy')
train_label_temp = np.load('label_Joe_aug.npy')
print('label 1: ', train_label_temp[0][0])
#
# train_label_Joe_temp = np.append(label1, label2, axis=0)
# train_label_Joe_temp = np.append(train_label_Joe_temp, label3, axis=0)
# train_label_Joe_temp = np.append(train_label_Joe_temp, label4, axis=0)
# # train_label_Joe_temp = train_label_Joe_temp.reshape(-1, 4)
# print('train data joe shape: ', train_data_Joe.shape)
# print('train label joe temp shape: ', train_label_Joe_temp.shape)
# # print(train_label_Joe[0])
#
train_label_x = np.empty((len(train_data)), np.float64)
train_label_y = np.empty((len(train_data)), np.float64)
err = 0
# print(train_label_Joe_temp[1][1])
#
for i in range(len(train_label_temp)):
    # print(train_label_temp[i][0])
    # k = train_label_temp[i][0]
    train_label_x[i] = train_label_temp[i][0]
    train_label_y[i] = train_label_temp[i][1]

# print('train label shape: ', train_label.shape)

########################################################################################################################
#setting training sets
########################################################################################################################

#here should be a normalization step

# training_set, val_set, training_label_x, val_label_x = train_test_split(train_data, train_label_x, test_size=0.2, random_state=42)
# training_set, val_set, training_label_y, val_label_y = train_test_split(train_data, train_label_y, test_size=0.2, random_state=42)

training_set = torch.from_numpy(train_data)
# print(training_set.dtype)
training_label_x = torch.from_numpy(train_label_x).type(torch.LongTensor)
training_label_y = torch.from_numpy(train_label_y).type(torch.LongTensor)
# training_label = training_set.to(torch.float32)

train = torch.utils.data.TensorDataset(training_set, training_label_x, training_label_y)


LR = 0.001
batch_size = 16 # smaller
num_epochs = 100


########################################################################################################################
#testing
########################################################################################################################
# batch = 1
test_data = np.load('data_Han_aug.npy')
test_label = np.load('label_Han_aug.npy')
print('test label shape: ', test_label.shape)
# test_label_Han_temp = test_label_Han_temp.reshape(-1, 4)
# print(len(test_label_Han_temp))

test_label_x = np.empty((len(test_data)), np.float64)
test_label_y = np.empty((len(test_data)), np.float64)
err = 0
# print(train_label_Joe_temp[1][1])
#
for i in range(len(test_label)):
    # print(train_label_temp[i][0])
    # k = train_label_temp[i][0]
    test_label_x[i] = test_label[i][0]
    test_label_y[i] = test_label[i][1]


# test_label_Han = np.empty((len(test_label_Han_temp), ), float)
#
# for i in range(len(test_label_Han_temp)):
#     test_label_Han[i] = (test_label_Han_temp[i][1] - 1)*24 + test_label_Han_temp[i][0]
#     if test_label_Han[i] < 0:
#         test_label_Han[i] = 0

testing_data = torch.from_numpy(test_data)
testing_label_x = torch.from_numpy(test_label_x).type(torch.LongTensor)  # data type is long
testing_label_y = torch.from_numpy(test_label_y).type(torch.LongTensor)
# print('testing data shape: ', testing_data.shape)
# print('testing label shape: ', testing_label.shape)

testing = torch.utils.data.TensorDataset(testing_data, testing_label_x, testing_label_y)
# testing_loader = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)



train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=True)


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
        self.fc1 = nn.Linear(32 * 6 * 4, 32)
        self.fc2 = nn.Linear(32 * 6 * 4, 24)
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
        out_temp = out.view(out.size(0), -1)
        # print('x7 shape: ', out.shape)
        # Linear function (readout)
        out_x = self.fc1(out_temp)
        # print('out_x type: ', type(out_x))
        out_y = self.fc2(out_temp)
        # print('out_y type: ', type(out_y))
        # print('out1 type: ', type((out_x, out_y)))

        # out = np.asarray(out_x, out_y)
        # out = torch.tensor(out)
        # print('out2 type: ', type(out))

        # print('x8 shape: ', out.shape)
        return out_x, out_y

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

correct_train = 0

def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
    # Training the Model
    #history-like list for store loss & acc value
    training_loss = []
    training_accuracy = []
    testing_loss = []
    testing_accuracy = []
    counter = 0
    best_score = None
    delta = 0.001
    early_stopping = None

    # if early_stopping == True:
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for images, labels_x, labels_y in train_loader:
            # print('1. ', images.shape)
            # print('2. ', labels.shape)
            # 1.Define variables
            train = Variable(images.view(input_shape))
            # print('train shape: ', train.shape)
            # print('train data type: ', train.dtype)
            labels_x = Variable(labels_x)
            labels_y = Variable(labels_y)
            # print('labels shape: ', labels.shape)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            (output_x, output_y) = model(train)
            # print('output_x shape: ', output_x.shape)
            # print('output_y shape: ', output_y.shape)
            # 4.Calculate softmax and cross entropy loss
            train_loss_x = loss_func(output_x, labels_x)
            train_loss_y = loss_func(output_y, labels_y)
            train_loss = (train_loss_x + train_loss_y) / 2
            # print('train loss x: ', train_loss_x)
            # print('train loss y: ', train_loss_y)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted_x = torch.max(output_x.data, 1)[1]
            # print('predicted x: ', predicted_x)
            predicted_y = torch.max(output_x.data, 1)[1]
            # print('predicted y: ', predicted_y)
            # 8.Total number of labels
            total_train += len(labels_x)
            # 9.Total correct predictions
            # correct_train += (predicted_x == labels_x).float().sum()          # problem
        #10.store val_acc / epoch
        # train_accuracy = 100 * correct_train / float(total_train)             # problem
        # training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for images, labels_x, labels_y in test_loader:
            # 1.Define variables
            test = Variable(images.view(input_shape))
            # 2.Forward propagation
            outputs_x, outputs_y = model(test)
            # 3.Calculate softmax and cross entropy loss
            test_loss_x = loss_func(outputs_x, labels_x)
            test_loss_y = loss_func(outputs_y, labels_y)
            test_loss = (test_loss_x + test_loss_y) / 2
            # 4.Get predictions from the maximum value
            predicted_x = torch.max(outputs_x.data, 1)[1]
            predicted_y = torch.max(outputs_y.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels_x)
            # 6.Total correct predictions
            # correct_test += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        # val_accuracy = 100 * correct_test / float(total_test)         # problem
        # validation_accuracy.append(val_accuracy)                      # problem
        # 11.store val_loss / epoch
        testing_loss.append(test_loss.data)
        # print('Train Epoch: {}/{} Training_Loss: {:.4f} Training_acc: {:.6f}% Val_Loss: {:.4f} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
        print('Train Epoch: {}/{} Training_Loss: {:.4f} Val_Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss.data, test_loss.data))
    # return training_loss, training_accuracy, validation_loss, validation_accuracy

        ################################################################################################################
        #early stopping
        ################################################################################################################

        if best_score == None or test_loss < best_score:
            best_score = test_loss
        elif test_loss > best_score:
            counter += 1
            print('counter: ', counter)
        elif counter >= 5:
            early_stopping = True

    return training_loss, testing_loss

# training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader)
training_loss, testing_loss = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader)





########################################################################################################################



plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
plt.plot(range(num_epochs), testing_loss, 'g-', label='Testing_loss')
plt.title('Training & Testing loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
# plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
# plt.title('Training & Validation accuracy')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()




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
# # batch = 1
# test_data = np.load('data_Han_aug.npy')
# test_label = np.load('label_Han_aug.npy')
# print('test label shape: ', test_label.shape)
# # test_label_Han_temp = test_label_Han_temp.reshape(-1, 4)
# # print(len(test_label_Han_temp))
#
# test_label_x = np.empty((len(test_data)), np.float64)
# test_label_y = np.empty((len(test_data)), np.float64)
# err = 0
# # print(train_label_Joe_temp[1][1])
# #
# for i in range(len(test_label)):
#     # print(train_label_temp[i][0])
#     # k = train_label_temp[i][0]
#     test_label_x[i] = test_label[i][0]
#     test_label_y[i] = test_label[i][1]
#
#
# # test_label_Han = np.empty((len(test_label_Han_temp), ), float)
# #
# # for i in range(len(test_label_Han_temp)):
# #     test_label_Han[i] = (test_label_Han_temp[i][1] - 1)*24 + test_label_Han_temp[i][0]
# #     if test_label_Han[i] < 0:
# #         test_label_Han[i] = 0
#
# testing_data = torch.from_numpy(test_data)
# testing_label_x = torch.from_numpy(test_label_x).type(torch.LongTensor)  # data type is long
# testing_label_y = torch.from_numpy(test_label_y).type(torch.LongTensor)
# # print('testing data shape: ', testing_data.shape)
# # print('testing label shape: ', testing_label.shape)
#
# testing = torch.utils.data.TensorDataset(testing_data, testing_label_x, testing_label_y)
# testing_loader = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)
#
#
#
# counter = 0
# err_sum = 0
# predictlist_x = []
# predictlist_y = []
# labellist_x = []
# labellist_y = []
# testing_result = np.empty((100, 1, 32, 24), float)
# labeling_result_x = np.empty((100), float)
# labeling_result_y = np.empty((100), float)
# predicted_result_x = np.empty((100), float)
# predicted_result_y = np.empty((100), float)
#
# for images, labels_x, labels_y in (testing_loader):
#     test_test = Variable(images.view(input_shape))
#     labels_x = Variable(labels_x)
#     labels_y = Variable(labels_y)
#
#     outputs_x, outputs_y = model(test_test)
#
#     test_loss_x = loss_func(outputs_x, labels_x)
#     test_loss_y = loss_func(outputs_y, labels_y)
#     test_loss = (test_loss_x + test_loss_y) / 2
#
#     predicted_x = torch.max(outputs_x.data, 1)[1]
#     predicted_y = torch.max(outputs_y.data, 1)[1]
#
#
#
#     test_test = test_test.numpy()  # new testing
#     predicted_x = predicted_x.numpy()
#     predicted_y = predicted_y.numpy()
#     labels_x = labels_x.numpy()
#     labels_y = labels_y.numpy()
#
#     # test_loss.append(test_loss.data)
#
#     if (counter <= 9):
#         print('label_x: ', labels_x[:10])
#         print('prediction_x: ', predicted_x[:10])
#         print('label_y: ', labels_y[:10])
#         print('prediction_y: ', predicted_y[:10])
#         # c = 0
#         for i in range(10):
#             testing_result[counter*10 + i] = test_test[i]
#             predicted_result_x[counter*10 + i] = predicted_x[i]
#             labeling_result_x[counter*10 + i] = labels_x[i]
#             predicted_result_y[counter * 10 + i] = predicted_y[i]
#             labeling_result_y[counter * 10 + i] = labels_y[i]
#
#             predictlist_x.append(predicted_x[i])     # old
#             predictlist_y.append(predicted_y[i])
#             labellist_x.append(labels_x[i])
#             labellist_y.append(labels_y[i])
#
#         err_counter = 0
#
#         for i in range(10):
#             if labels_x[i] != predicted_x[i] or labels_y[i] != predicted_y[i]:
#                 err_counter += 1
#                 err_sum += 1
#
#         print('numbers of error: ', err_counter)
#
#     counter += 1
#
#
#
# ########################################################################################################################
# #confusion matrix
# ########################################################################################################################
#
# testing_result = testing_result.squeeze(1)
# print('testing result shape: ', testing_result.shape)
#
#
# print('total error: ', err_sum)
# print('confusion matrix: ')
# print(confusion_matrix(labellist_x, predictlist_x))
# print(confusion_matrix(labellist_y, predictlist_y))
#
# print('testing accuracy: ', (100 - err_sum) / 100 * 100, '%')
# ########################################################################################################################
# # visualization
# ########################################################################################################################
#
# # plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
# # plt.plot(range(num_epochs), testing_loss, 'g-', label='testing_loss')
# # # plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
# # # plt.title('Training & Validation loss')
# # plt.title('Training & Testing loss')
# # plt.xlabel('Number of epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
# #
# # plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
# # plt.plot(range(num_epochs), testing_accuracy, 'g-', label='Validation_accuracy')
# # # plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
# # # plt.title('Training & Validation accuracy')
# # plt.title('Training & Testing accuracy')
# # plt.xlabel('Number of epochs')
# # plt.ylabel('Accuracy')
# # plt.legend()
# # plt.show()
#
# ########################################################################################################################
#
# for i in range(10):
#     label_2 = labeling_result_y[i]
#     label_1 = labeling_result_x[i]
#     predicted_2 = predicted_result_y[i]
#     predicted_1 = predicted_result_x[i]
#
#     fig, ax = plt.subplots()
#     ax.imshow(testing_result[i], vmin=22, vmax=32.5, cmap='jet')
#
#     fig, ax = plt.subplots()
#     ax.imshow(testing_result[i], vmin=22, vmax=32.5, cmap='jet')
#     rect1 = patches.Rectangle(((label_1 - 2), (label_2 - 2)), 5, 5, lw=2, ec='r', fc='none')
#     ax.add_patch(rect1)
#     rect2 = patches.Rectangle(((predicted_1 - 2), (predicted_2 - 2)), 5, 5, lw=2, ec='y', fc='none')
#     ax.add_patch(rect2)
#
# ########################################################################################################################
#
#     # ax.imshow(testing_result[i], vmin=22, vmax=32.5, cmap='jet')
#     # rect1 = patches.Rectangle((label_1, label_2), 5, 5, lw=2, ec='r', fc='none')
#     # ax.add_patch(rect1)
#     # rect2 = patches.Rectangle((predicted_1, predicted_2), 5, 5, lw=2, ec='y', fc='none')
#     # ax.add_patch(rect2)
#
#
#
# plt.show()