import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(RNN, self).__init__()
        self.w1 = nn.Linear(in_size, hidden_size)
        self.w2 = nn.Linear(hidden_size,4) # Output 4
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        # self.cuda()

    def forward(self,x):
        z2 = self.w1(x)
        a2 = self.softmax(z2)
        z3 = self.w2(a2)
        # return z3
        return self.softmax(z3)

    
if __name__ == '__main__':
    net = RNN(2, 15)
    net.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.0005)
    train = [[1,2],[1,-1],[-5,3],[-4,-4],[5,2],[-4,10]] # TODO: load input (expected keywords from headlines)
    train_label = [[0],[3],[1],[2],[0],[1]] # TODO: load input label

    epochs = 5000
    for _ in range(epochs):
        for i in range(len(train)):
            x = torch.tensor(np.array(train[i],dtype=np.float64), requires_grad=True)
            x.float()
            y = torch.tensor(train_label[i])
            y.long()
            # print(x)
            # print(y)

            optimizer.zero_grad()
            pred = net.forward(x) # Get prediction result
            pred = pred.view((1,4))
            # print(pred)

            loss = criterion(pred,y) # Loss function
            loss.backward()
            optimizer.step()
            # print("PASS %d" % i)

    test = torch.tensor(np.array([30,20], dtype=np.float64))
    test.float()
    print(net(test))