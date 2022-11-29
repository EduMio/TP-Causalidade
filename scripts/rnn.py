import numpy as np
import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        
        self.n_attributes = 146
        self.n_targets = 129

        self.rnn = nn.RNN(input_size = self.n_attributes,hidden_size = self.n_targets,num_layers = 10,dropout = 0.3)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

        
    def forward(self, x):
        
        [x,hn] = self.rnn(x)
        return x
