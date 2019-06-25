#! /usr/bin/env python3

########################################################
#########      Main file for this project      #########
########################################################

import numpy as np
from RNN import RNN
# import matplotlib.pyplot as plt # For plotting later

net = RNN(in_size=500,out_size=2)

train = None # TODO: Load train data of size nx500
train_label = None # TODO: Load train data label of size nx1
 
# Param: X,Y,epoch=100,lr=0.0005
net.fit(train,train_label)

test = None # TODO: Load test data of size nx500
test_label = None # TODO: Load test data label of size nx1

acc = [0 for _ in range(len(test))] # Accuracy: 1 is correct, 0 is incorrect

# 0 for going down and 1 for going up
for i in range(len(test)):
    result = net(test[i])[0]
    if result[0].item() < result[1].item(): # Predict to rise
        acc[i] = int(1==int(test_label[i])) # Update to 1 if predict correctly
    else: # Predict to fall
        acc[i] = int(0==int(test_label[i])) # Update to 1 if predict correctly

np.save('Accuracy',acc) # For process later
print(np.average(acc)) # Print out accuracy of prediction

# TODO: Plot accuracy-epoch?