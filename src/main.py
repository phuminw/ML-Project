#! /usr/bin/env python3

########################################################
#########      Main file for this project      #########
########################################################

import numpy as np
import pandas as pd
from RNN import RNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt # For plotting later

# Load data from preprocessed csv
def load_data(name="MSFT/news_and_change.csv", today=True):
    nc = pd.read_csv(name)
    headlines = nc['TITLE'].values
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(headlines) # TD-IDF

    if today:
        return X.toarray(), nc['today'].values
    else:
        return X.toarray(), nc['tomorrow'].values

net = RNN(in_size=500,hidden_size=50,out_size=2)

# K-Fold CV
kf = KFold(n_splits=10, random_state=None, shuffle=False)
# Load data and label
X, y = load_data()

acc = []
for i in range(100):
    acc_local = [] # To store accuracy of k times from k-Fold
    for train_index, test_index in kf.split(X):
        # Load train and test data and label
        train, train_label, test, test_label = X[train_index],y[train_index] ,X[test_index], y[test_index]
    
        # Param: X,Y,epoch=100,lr=0.0005
        net.fit(train,train_label,epoch=50)

        acc_fold = 0 # Counter for correct prediction

        for j in range(len(test)):
            pred = net(test[j])[0]
            acc_fold += int((pred[0]<pred[1])==int(test_label[j])) # Accumulate # of correct prediction

        # Log accuracy of each fold
        acc_local.append(acc_fold/len(test))

    acc.append(np.average(acc_local)) # Accuracy of 1 epoch



# 0 for going down and 1 for going up
# for i in range(len(test)):
#     result = net(test[i])[0]
#     if result[0].item() < result[1].item(): # Predict to rise
#         acc[i] = int(1==int(test_label[i])) # Update to 1 if predict correctly
#     else: # Predict to fall
#         acc[i] = int(0==int(test_label[i])) # Update to 1 if predict correctly

np.save('Accuracy',acc) # For process later
print(np.average(acc)) # Print out accuracy of prediction

# TODO: Plot accuracy-epoch?