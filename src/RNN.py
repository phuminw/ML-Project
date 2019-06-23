#! /usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(RNN, self).__init__()
        self.w1 = nn.Linear(in_size, hidden_size)
        self.w2 = nn.LSTM(hidden_size,hidden_size,2,dropout=0.05)
        self.w3 = nn.Linear(hidden_size,4) # Output 4
        self.sigmoid = nn.Sigmoid() 
        self.softmax = nn.Softmax(dim=0) # For output as probability
        self.cuda() # Use GPU

    def forward(self,x,hidden):
        x = x.double().view((1,len(x))) # Preprocess x
        z2 = self.w1(x).cuda() # Input to hidden 
        a2 = self.sigmoid(z2).cuda() # Output of hidden
        z3, hid = self.w2(a2.unsqueeze(1), hidden) # Input to LSTM
        z3 = self.sigmoid(z3.squeeze(1)) # Output of LSTM
        return self.softmax(self.w3(z3.cuda()).view(4)), hid # Return prob of len(out) classes and hidden output for recurrent

    
if __name__ == '__main__':
    net = RNN(2, 15) # Init NN with 2 in and 15 in hidden
    net.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.0005)
    train = [[1,2],[1,-1],[-5,3],[-4,-4],[5,2],[-4,10]] # TODO: load input 
    train_label = [[0],[3],[1],[2],[0],[1]] # TODO: load input label

    epochs = 2000
    hidden=None
    for _ in range(epochs):
        for i in range(len(train)):
            x = torch.tensor(np.array(train[i],dtype=np.float64), requires_grad=True).cuda()
            y = torch.tensor(train_label[i]).cuda()
            y = y.long().cuda()

            optimizer.zero_grad()
            pred,hidden = net(x,hidden) # Get prediction result
            pred = pred.view((1,4)) # Prediction in appropiate shape

            loss = criterion(pred,y) # Loss function
            loss.backward(retain_graph=True)
            optimizer.step()

    test = torch.tensor(np.array([30,20], dtype=np.float64)).cuda() # For testing
    test = test.float().cuda()
    print(net(test,hidden)[0])