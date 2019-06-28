#! /usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, in_size=500, hidden_size=15, out_size=4):
        super(RNN, self).__init__()
        self.in_size = in_size
        self.out_size= out_size
        self.w1 = nn.Linear(in_size, hidden_size)
        self.w2 = nn.LSTM(hidden_size,hidden_size,2,dropout=0.05)
        self.w3 = nn.Linear(hidden_size,out_size) # To output weight
        self.sigmoid = nn.Sigmoid() 
        self.softmax = nn.Softmax(dim=0) # For output as probability
        self.cuda() # Use GPU

        self.hidden = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None


    # Expect 500 real-valued input (shape nx10) as list/ndarray both X and y
    def fit(self,X,Y,epoch=100,lr=0.0005):
        try:
            # Data type assertion
            assert(((type(X) == list) or (type(X) == np.ndarray)) and ((type(Y) == list) or (type(Y) == np.ndarray)))

            # Reshape to ensure shape
            X = torch.tensor(X, dtype=torch.float64, requires_grad=True).view((len(X),self.in_size)).cuda()
            Y = torch.tensor(Y, dtype=torch.float64, requires_grad=True).view((len(Y),1)).cuda() 

            self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)

            for _ in range(epoch):
                for i in range(X.shape[0]):
                    self.optimizer.zero_grad()
                    pred,self.hidden = self.forward(X[i].float()) # Feed forward
                    pred = pred.view((1,self.out_size)) # Prediction in appropiate shape

                    loss = self.criterion(pred,Y[i].long()) # Loss function
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
            return True,self.hidden # Finished training and return memory
        except AssertionError:
            print('Input type should be either list or ndarray.')


    def forward(self,x):
        x = x.view((1,len(x))) # Preprocess x
        z2 = self.w1(x).cuda() # Input to hidden 
        a2 = self.sigmoid(z2) # Output of hidden
        z3, hid = self.w2(a2.unsqueeze(1), self.hidden) # Input to LSTM
        return self.softmax(self.w3(z3.squeeze(1)).view(self.out_size)), hid # Return prob of len(out) classes and hidden output for recurrent

# Save model for later use
def save_model(model, name):
    try:
        torch.save(model, name)
        return True
    except:
        return False

# Load model from file
def load_model(name):
    try:
        return torch.load(name)
    except:
        return None
    
# For testing purpose
if __name__ == '__main__':
    net = RNN() # Init NN with 2 in and 15 in hidden
    # net.double() # Not necessary (Maybe)
    train = [[1,2],[1,-1],[-5,3],[-4,-4],[5,2],[-4,10]] # TODO: load input 
    train_label = [[0],[3],[1],[2],[0],[1]] # TODO: load input label

    net.fit(train,train_label) # Train with data

    test = torch.tensor([30,20]).cuda() # For testing
    test = test.float() # Must call .float before prediction
    print(net(test)[0])