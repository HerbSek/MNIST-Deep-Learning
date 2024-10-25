import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.model = nn.Sequential(
               nn.Conv2d(1,16,3,1,1),
               nn.ReLU(),
               nn.MaxPool2d(2,2),

               nn.Conv2d(16, 32, 3,1,1),
               nn.ReLU(),
               nn.MaxPool2d(2,2),
              
           )
        self.linear1 =  nn.Linear(32*7*7, 128)
        self.linear2 = nn.ReLU()
        self.linear3 = nn.Linear(128,10)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 32*7*7)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x