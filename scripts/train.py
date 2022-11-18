import numpy as np
import torch
import torch.nn as nn
import time

import linear_model
import utils

# Hyperparameters

n_epochs = 2
batch_size = 2000

n_attributes = 146
n_targets = 129

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Loading the data

X_train, Y_train, X_test, Y_test = utils.get_data(path = "../data/training/db.npy",device = device,split_train_test = 1,n_attributes = n_attributes)


# Declaring the model,criterion and optmizer

model = linear_model.LinearModel()
model.to(device)

#Training

for epoch in range(n_epochs):
    print(epoch)
    for i in range(0,len(X),batch_size):
        
        x_train = X[i:i+batch_size]
        y_train = Y[i:i+batch_size]
        
        model.optimizer.zero_grad()
        y_pred = model(x_train)
        loss = model.criterion(y_pred, y_train)
        loss.backward()
        model.optimizer.step()
        
    print(loss.to("cpu").detach().numpy())

utils.save_model(model,path = "../data/models/linear1.pth")
