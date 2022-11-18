import numpy as np
import torch
import torch.nn as nn

def save_model(model,path):
    torch.save(model.state_dict(), path)

def load_model(model_class):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))

    return model

def get_data(path,device,split_train_validation,n_attributes,train_fraction = 0.8):
    
    db = np.load(path)
    
    X = (db.T[:n_attributes]).T
    Y = (db.T[n_attributes:]).T

    del(db)
    
    if split_train_validation:
        split = np.random.choice(range(X.shape[0]), int(train_fraction*X.shape[0]))

        X_train = X[split]
        Y_train = Y[split]
        X_test =  X[~split]
        Y_test = Y[~split]
        
        X_train, Y_train, X_test, Y_test = torch.tensor(X_train).float(),torch.tensor(Y_train).float(),torch.tensor(X_test).float(),torch.tensor(Y_test).float()
        
        X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
        
        return X_train, Y_train, X_test, Y_test

    else:
        
        X,Y = torch.tensor(X).float(),torch.tensor(Y).float()
        X,Y = X.to(device),Y.to(device)
        
        return X,Y