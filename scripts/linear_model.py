import numpy as np
import torch
import torch.nn as nn

class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        
        self.n_attributes = 146
        self.n_targets = 129
        self.n_scalating = 5
        self.scalating_constant = 2 # Keep this a integer, float values can get things unstable
        self.n_linears_layers = 2*self.n_scalating+3 # MUST be equal to 2*n_scalating + 3 in this implementation: 
        #1 starting layer, n_scalating layers that the number of nodes increase, 1 intermediate layer, 
        # n_scalating layers that the number of nodes decrease, 1 final layer
        
        self.layers = nn.ModuleList([])
        self.create_layers()
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def create_layers(self):
        
        n_targets = self.n_targets
        n_linears_layers = self.n_linears_layers
        n_attributes = self.n_attributes
        n_scalating = self.n_scalating
        scalating_constant = self.scalating_constant
        
        last_n_nodes = self.scalating_constant*self.n_attributes
        
        self.layers.append(torch.nn.Linear(n_attributes, last_n_nodes))
        j = 0
        for i in range(1,n_linears_layers-n_scalating-1):
            self.layers.append(torch.nn.Linear(last_n_nodes, scalating_constant*last_n_nodes))
            last_n_nodes = scalating_constant*last_n_nodes
            j = i
        
        self.layers.append(torch.nn.Linear(last_n_nodes, last_n_nodes))
        
        for i in range(j,n_linears_layers-1):
            self.layers.append(torch.nn.Linear(last_n_nodes, int(last_n_nodes/scalating_constant)))
            last_n_nodes = int(last_n_nodes/scalating_constant)
        
        self.layers.append(torch.nn.Linear(last_n_nodes, n_targets))
        
    
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x
