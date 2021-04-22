# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:16:48 2021

@author: Nir
"""

# ==== Imports ====

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from scipy.stats import entropy

# ==== functions ====

def f_ising_creator(n, m):
  x = product([1, -1], repeat=n*m)
  x = np.reshape(list(x), (-1, n, m))
  for i in x:
    i[i==0] = -1
  lattice_combinations = x
  return lattice_combinations
# split our lattices into left and right ones
def f_splitter(lattice):
    if lattice.shape[1] > lattice.shape[2]:
        lattice = np.transpose(lattice, axes=(0,2,1))       
    left_lattices, right_lattices = np.split(lattice, 2, axis=2)    
    return left_lattices, right_lattices
# calculate energy lattice of nxm lattice
def f_entropy(lattices, J, kT):   
    energy = (lattices * J * (
                            np.roll(lattices, shift=1, axis=1) +
                            np.roll(lattices, shift=-1, axis=1) +
                            np.roll(lattices, shift=1, axis=2) +
                            np.roll(lattices, shift=-1, axis=2)
                            )).sum(axis=(1,2))
    boltzmann = np.exp(-energy/kT) # unnormalized boltzmann
    entropies = entropy(boltzmann)
    return entropies, energy  
# calculate mutual information
def f_mutual_informations(S_ab, S_a, S_b):   
  mutual_informations = (S_a + S_b) - S_ab 
  return mutual_informations
# ==== Main ====
def run_ising(kT, n, m, J):
  #kT = 3
  #J = 1
  #n = 2
  #m = 4
  lattices = np.array(f_ising_creator(n, m)) # list of nxm ising lattices
  left_lattices, right_lattices = f_splitter(lattices) # split all lattices into left and right ones
  entropy_2_4, energy_full = f_entropy(lattices, J, kT) # calculate the entropy of the big lattices
  entropy_2_2_left, energy_left = f_entropy(left_lattices, J, kT) # calculate the entropy of the small left lattices
  entropy_2_2_right, energy_right = f_entropy(right_lattices, J, kT) # calculate the entropy of the small right lattices
  mutual_information = f_mutual_informations( entropy_2_4, entropy_2_2_right, entropy_2_2_left) # calculate the mutual information
  return mutual_information
def main():

    list_mutual_information = []
    start_kT = 0.01
    end_kT = 10
    num_T = 100   
    J = 1   
    n= 2
    m = 4
    for kT in np.linspace(start_kT, end_kT, num=num_T, endpoint=False):
      mutual_information = run_ising(kT, n, m, J)
      list_mutual_information.append(mutual_information)  
    plt.plot(np.linspace(start_kT, end_kT, num=num_T, endpoint=False), list_mutual_information, linewidth=5)
    plt.title(f'{n}x{m} ising model',fontsize=25)
    plt.ylabel('Averaged mutual information',fontsize=15)
    plt.xlabel('kT', fontsize=15)
    
    
    
# ==========================================



#main()