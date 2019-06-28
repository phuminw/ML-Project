#! /usr/bin/env python3

########################################################
#########      Main file for this project      #########
########################################################

import numpy as np
import pandas as pd
import torch
from RNN import RNN,save_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt # For plotting later

# Load data from preprocessed csv
def load_data(name="AMZN/AMZN_filtered_labeled.csv", today=True):
    nc = pd.read_csv(name).sample(100)
    headlines = nc['TITLE'].values
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(headlines) # TD-IDF

    if today:
        return X.toarray(), nc['today'].values
    else:
        return X.toarray(), nc['tomorrow'].values

net = RNN(in_size=10,hidden_size=4,out_size=2)

# K-Fold CV
kf = KFold(n_splits=2, random_state=None, shuffle=False)
# Load data and label
X, y = load_data()

acc = []
for i in range(10):
    acc_local = [] # To store accuracy of k times from k-Fold
    for train_index, test_index in kf.split(X):
        # Load train and test data and label
        train, train_label, test, test_label = X[train_index],y[train_index] ,X[test_index], y[test_index]
    
        # Param: X,Y,epoch=100,lr=0.0005
        net.fit(train,train_label,epoch=1,lr=0.0005)

        acc_fold = 0 # Counter for correct prediction

        for j in range(len(test)):
            to_feed = torch.tensor(test[j], dtype=torch.float64, requires_grad=True).cuda()
            pred = net(to_feed.float())[0]
            acc_fold += int((pred[0]<pred[1])==int(test_label[j])) # Accumulate # of correct prediction

        # Log accuracy of each fold
        acc_local.append(acc_fold/len(test))

    acc.append(np.average(acc_local)) # Accuracy of 1 epoch
    with open('log/{}'.format(i), 'w') as f:
        f.write("Done epoch {}".format(i))


# 0 for going down and 1 for going up
# for i in range(len(test)):
#     result = net(test[i])[0]
#     if result[0].item() < result[1].item(): # Predict to rise
#         acc[i] = int(1==int(test_label[i])) # Update to 1 if predict correctly
#     else: # Predict to fall
#         acc[i] = int(0==int(test_label[i])) # Update to 1 if predict correctly


np.save('Accuracy_AMZN',acc) # For process later
print(np.average(acc)) # Print out accuracy of prediction
save_model(net, 'AMZN_small.model')

# TODO: Plot accuracy-epoch?