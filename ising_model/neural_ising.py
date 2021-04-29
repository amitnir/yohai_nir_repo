# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:46:40 2021
@author: Nir
"""

import simple_ising as si
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

# Deep neural network
# Here I define the model as was written in the MICE article

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(2*4, 2*1)
      self.fc2 = nn.Linear(2*1, 1)

    def forward(self, x):
      x = x.view(-1, 2*4) # turn it into a 2d tensor
      x = self.fc1(F.relu(x))
      x = self.fc2(F.relu(x))
      #x = self.fc1(x)
      #x = self.fc2(x)
      return x
    
def loss_function(joint_prob, product_prob, joint_output, product_output):
    joint_prob = joint_prob.reshape(-1,1)
    product_prob = product_prob.reshape(-1,1)
    joint_part = (joint_prob * joint_output).sum()
    product_part = torch.log(
                         (product_prob * (torch.exp(product_output))).sum()
                         )
    mutual = joint_part - product_part
    
    return mutual

def run_ising(kT, n, m, J):
    lattices = np.array(si.f_ising_creator(n, m)) # list of nxm ising lattices
    left_lattices, right_lattices = si.f_splitter(lattices) # split all lattices into left and right ones
    return lattices, left_lattices, right_lattices
    
def calcualte_entropy_for_one_temperature(kT):
    print('here')
    #kT = 9999
    J = 1
    n= 2
    m = 4 
    batch_size = 256
    lattices, left_lattices, right_lattices = run_ising(kT, n, m, J)
    AB = torch.tensor(lattices)
    loader = DataLoader(AB, batch_size=batch_size, shuffle=False)
    model = Net()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    n_epochs = 20
    # empty list to store training losses
    train_losses = []
    joint_prob, product_prob = si.f_prob_calc(J, kT, n, m)
    joint_prob = torch.tensor(joint_prob, requires_grad=False)
    product_prob = torch.tensor(product_prob, requires_grad=False)
    # training the model
    for epoch in range(n_epochs):
        flag = 0
        for batch_idx, x in enumerate(loader):
          # x is (256, 2, 4) shape. it is the lattices tensor
            optimizer.zero_grad()
            joint_output = model(x.float())
            product_output = model(x.float())
            loss_train = loss_function(joint_prob[flag:flag+batch_size], product_prob[flag:flag+batch_size], joint_output[flag:flag+batch_size], product_output[flag:flag+batch_size])
            flag += batch_size   
            # computing the updated weights of all the model parameters          
            loss_train.backward()
            optimizer.step()
        train_losses.append(loss_train)
        print(train_losses[-1])
    return train_losses[-1]     

train_losses = calcualte_entropy_for_one_temperature(kT=1)