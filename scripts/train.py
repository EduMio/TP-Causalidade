import numpy as np
import torch
import torch.nn as nn
import time

# Time Series Transformer
# ResNets

import linear_model
from rnn import RNN
import utils, os

# Hyperparameters

n_epochs = 100 # Total = 100
batch_size = 300

n_attributes = 146
n_targets = 129

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_model = 1

training = 0

#Loading the data

#X_train, Y_train, X_test, Y_test = utils.get_data(path = "../data/training/db.npy",device = device,split_train_validation = 1,n_attributes = n_attributes)

DIR = '../data/training/'
PATH = DIR + 'db.npy'
if not os.path.exists(PATH):
    print('Concatenando todos os dados de treino')
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    def getV(i):
        return np.load('../preprocess/preprocessed' + str(i) + '.npy')
    v = getV(0)
    for i in range(1, 18):
        aux = getV(i)
        v = np.concatenate((v, aux))
    np.save(PATH, v)

if training:
    X,Y = utils.get_data(path = PATH,device = device,split_train_validation = 0,n_attributes = n_attributes)
else:
    
    
# Declaring the model,criterion and optmizer

if load_model:
    model = RNN()
    model.load_state_dict(torch.load("../data/models/rnn/rnn_1_epoch45.pth"))
    model.eval()
    model.to(device)
    print("model loaded")
else:
    model = RNN()
    model.to(device)
    print("model loaded")

    
if training:

#Training

    start_training = time.time()
    for epoch in range(50,n_epochs): #Training interrupted in  epoch 20
        start_epoch = time.time()
        if epoch%5 == 0 and epoch > 50:
            utils.save_model(model,path = "../data/models/rnn/rnn_1_epoch"+str(epoch)+".pth")
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


    utils.save_model(model,path = "../data/models/rnn/rnn_1.pth")
else:
    Y_pred = []
    for i in range(0,len(X),batch_size):

        x = X[i:i+batch_size]

        y_pred = model(x)
        Y_pred.append(y_pred)