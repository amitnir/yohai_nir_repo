# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:46:40 2021

@author: Nir
"""

import simple_ising
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Deep neural network
# Here I define the model as was written in the MICE article

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(128, 64)
      self.fc2 = nn.Linear(64, 16)
      self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
      x = x.view(-1, x.size(0)) # turn it into a 2d tensor
      x = self.fc1(F.relu(x))
      x = self.fc2(F.relu(x))
      x = self.fc3(F.relu(x))
      return x
    
def LossFunction(f_joint, f_product, J, kT):
    E_joint = (f_joint * J * (
                            np.roll(f_joint, shift=1, axis=0) +
                            np.roll(f_joint, shift=-1, axis=0) +
                            np.roll(f_joint, shift=1, axis=1) +
                            np.roll(f_joint, shift=-1, axis=1)
                            )).sum(axis=(0,1))
    boltzmann_joint = np.exp(-E_joint/kT) # unnormalized boltzmann 
    Z_joint = boltzmann_joint.sum() 
    mean_joint = (1/Z_joint) * ((boltzmann_joint * E_joint).sum())
    E_product = (f_product * J * (
                            np.roll(f_product, shift=1, axis=0) +
                            np.roll(f_product, shift=-1, axis=0) +
                            np.roll(f_product, shift=1, axis=1) +
                            np.roll(f_product, shift=-1, axis=1)
                            )).sum(axis=(0,1))
    boltzmann_product = np.exp(-E_product/kT) # unnormalized boltzmann 
    Z_product = boltzmann_product.sum()
    mean_product = (1/Z_product) * ((boltzmann_product * np.exp(E_product)).sum()) 
    return -(mean_joint - np.log(mean_product))

def run_ising(kT, n, m, J):
    lattices = np.array(simple_ising.f_ising_creator(n, m)) # list of nxm ising lattices
    left_lattices, right_lattices = simple_ising.f_splitter(lattices) # split all lattices into left and right ones
    return left_lattices, right_lattices

def main():
    kT = 0.5 
    J = 1
    n= 2
    m = 4 
    seed = 1234
    RS = np.random.RandomState(seed)
    left_lattices, right_lattices = run_ising(kT, n, m, J)
    right_lattices_copy = copy.deepcopy(right_lattices)
    A_joint = torch.tensor(left_lattices)
    B_joint = torch.tensor(right_lattices)
    A_product = torch.tensor(left_lattices)
    RS.shuffle(right_lattices_copy)
    B_product = torch.tensor(copy.deepcopy(right_lattices_copy)) 
    # ==== AB_joint and AB_product are 256x2x4 shape ====  
    AB_joint = torch.tensor(np.concatenate((A_joint, B_joint), axis=2)).unsqueeze(3)
    AB_product = torch.tensor(np.concatenate((A_product, B_product), axis=2)).unsqueeze(3)
    AB = torch.tensor(np.concatenate((AB_joint, AB_product), axis=3))
    AB = AB.unsqueeze(1)
    loader = DataLoader(AB, batch_size=128, shuffle=True)
    model = Net()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    n_epochs = 25
    # empty list to store training losses
    train_losses = []
    print("\n==== The model architecture ====\n")
    print(model)
    print()
    # training the model
    flag = 0
    for epoch in range(n_epochs):
        for batch_idx, x in enumerate(loader):
          # train the model
          model.train()
          # changing the elements from torch elements into numpy elements
          joint_output = model(x[:,:,:,0].float())
          product_output = model(x[:,:,:,1].float())
          print(joint_output.shape)
          loss_train = torch.tensor(LossFunction(joint_output.detach().numpy(), product_output.detach().numpy(), J, kT))
          loss_train.requires_grad=True     
          # computing the updated weights of all the model parameters
          optimizer.zero_grad() 
          loss_train.backward()
          optimizer.step()
          if flag == 0:
              flag = 1
              print("==== unit test ====\n")
              print(f"AB joint shape is: {AB_joint.shape}")
              print(f"AB product shape is: {AB_product.shape}")
              print(f"AB shape is: {AB.shape}")
              print(f"The shape of the joint model's input is: {x[:,:,:,0].float().shape}")
              print(f"The shape of the product model's input is: {x[:,:,:,1].float().shape}")
              print(f"The shape of the joint model's output is: {joint_output.shape}")
              print(f"The shape of the product model's output is: {product_output.shape}")
              print()
              print("epoch : loss")
              print("=============")
              print()
        train_losses.append(loss_train)
        print(epoch,' : ',loss_train.item())
    return train_losses      
train_losses = main()
