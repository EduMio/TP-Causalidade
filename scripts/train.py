import numpy as np
import torch
import torch.nn as nn
import time

import linear_model
import utils

# Hyperparameters

n_epochs = 50
batch_size = 1500

n_attributes = 146
n_targets = 129

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Loading the data

#X_train, Y_train, X_test, Y_test = utils.get_data(path = "../data/training/db.npy",device = device,split_train_validation = 1,n_attributes = n_attributes)

X,Y = utils.get_data(path = "../data/training/db.npy",device = device,split_train_validation = 0,n_attributes = n_attributes)

# Declaring the model,criterion and optmizer

model = linear_model.LinearModel()
model.to(device)

#Training


start_training = time.time()
for epoch in range(n_epochs):
    start_epoch = time.time()
    if epoch%5 == 0 and epoch > 0:
        utils.save_model(model,path = "../data/models/lin_5_layers/linear_5_scalating_epoch_"+str(epoch)+".pth")
    for i in range(0,len(X),batch_size):
        
        x = X[i:i+batch_size]
        y = Y[i:i+batch_size]
        
        model.optimizer.zero_grad()
        y_pred = model(x)
        loss = model.criterion(y_pred, y)
        loss.backward()
        model.optimizer.step()
    
    time_epoch = (time.time() - start_epoch)/60
    
    print("Epoch: "+str(epoch),"Time spent (Minutes): "+str(time_epoch),"Loss: "+str(loss.to("cpu").detach().numpy()),sep="\n")
    print("--------------------------------------------------------------------\n")
    
time_training = (time.time() - start_training)/3600
print("Time spent (Hours): "+str(time_training),"Avg time spent per epoch (Minutes) : "+str(60*time_training/n_epochs),"Final Loss: "+str(loss.to("cpu").detach().numpy()),sep="\n")


utils.save_model(model,path = "../data/models/lin_5_layers/linear_5_scalating.pth")
