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
# calculate boltzmann normalized probabilities
def f_boltzmann(lattice, J, kT):   
    energy = (lattice * J * (
                            np.roll(lattice, shift=1, axis=0) +
                            np.roll(lattice, shift=-1, axis=1) +
                            np.roll(lattice, shift=1, axis=0) +
                            np.roll(lattice, shift=-1, axis=1)
                            )).sum(axis=(0,1))
    boltzmann = np.exp(-(energy)/kT) # unnormalized boltzmann
    return boltzmann
def f_prob_calc(J, kT, n, m):   
    left_comb = f_ising_creator(n,int(m/2))
    lattices = np.array(f_ising_creator(n, m))
    left_lattices, right_lattices = f_splitter(lattices)
    my_dict = {str(lattice): 0 for lattice in left_comb}
    joint_prob = np.array([f_boltzmann(lattice, J, kT) for lattice in lattices])
    joint_prob /= joint_prob.sum()
    flag = 0
    for left_lattice in left_lattices:
        my_dict[str(left_lattice)] += joint_prob[flag]
        flag += 1
    product_prob = np.array([my_dict[str(left_lattice)] * my_dict[str(right_lattice)]
                             for left_lattice in left_lattices
                             for right_lattice in right_lattices])
    return joint_prob, product_prob
# calculate left and right entropy
def f_prob_calc_left(J, kT, n, m):   
    left_comb = f_ising_creator(n,int(m/2))
    lattices = np.array(f_ising_creator(n, m))
    left_lattices, right_lattices = f_splitter(lattices)
    my_dict = {str(lattice): 0 for lattice in left_comb}
    joint_prob = np.array([f_boltzmann(lattice, J, kT) for lattice in lattices])
    joint_prob /= joint_prob.sum()
    flag = 0
    for left_lattice in left_lattices:
        my_dict[str(left_lattice)] += joint_prob[flag]
        flag += 1
    prod_prob_left = np.array([prob for prob in my_dict.values()])
    return prod_prob_left
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
  #entropy_2_2_left, energy_left = f_entropy(left_lattices, J, kT) # calculate the entropy of the small left lattices
  #entropy_2_2_right, energy_right = f_entropy(right_lattices, J, kT) # calculate the entropy of the small right lattices
  product_prob_left = f_prob_calc_left(J, kT, n, m)
  entropy_2_2_left = entropy(product_prob_left)
  entropy_2_2_right = entropy_2_2_left
  mutual_information = f_mutual_informations( entropy_2_4, entropy_2_2_right, entropy_2_2_left) # calculate the mutual information
  return mutual_information
def theoretic_ising():

    list_mutual_information = []
    start_kT = 0.01
    end_kT = 10
    num_T = 100   
    J = 1   
    n= 2
    m = 4
    list_mutual_information = [
                               run_ising(kT, n, m, J)
                               for kT in np.linspace(start_kT, end_kT, num=num_T, endpoint=False)
                               ]
    plt.plot(np.linspace(start_kT, end_kT, num=num_T, endpoint=False), list_mutual_information, linewidth=5)
    plt.title(f'{n}x{m} ising model',fontsize=25)
    plt.ylabel('Averaged mutual information',fontsize=15)
    plt.xlabel('kT', fontsize=15)
    
    
    
# ==========================================



theoretic_ising()