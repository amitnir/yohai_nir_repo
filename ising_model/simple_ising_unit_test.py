# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:28:04 2021
@author: Nir
"""

import numpy as np
import simple_ising as si

# ==== input ==== #

n = 2
m = 4
J = 1
kT = 1
lat_num = 55

#=================#

def test_creator(n, m):
    
    lattices = np.array(si.f_ising_creator(n, m))
    assert lattices.shape==(2**(n*m), n, m)

def test_splitter(n, m):
      lattices = np.array(si.f_ising_creator(n, m))
      left_lattices, right_lattices = si.f_splitter(lattices) # split all lattices into left and right ones
      assert left_lattices.shape==(2**(n*m), n, int(m/2))
      assert right_lattices.shape==(2**(n*m), n, int(m/2))
      
      
def test_entropy_mutual(n, m):     
    lattices = np.array(si.f_ising_creator(n, m))
    left_lattices, right_lattices = si.f_splitter(lattices) # split all lattices into left and right ones
    entropy_2_4, energy_full = si.f_entropy(lattices, J, kT) # calculate the entropy of the big lattices
    entropy_2_2_left, energy_left = si.f_entropy(left_lattices, J, kT) # calculate the entropy of the small left lattices
    entropy_2_2_right, energy_right = si.f_entropy(right_lattices, J, kT) # calculate the entropy of the small right lattices
    mutual_information = si.f_mutual_informations(entropy_2_4, entropy_2_2_right, entropy_2_2_left) # calculate the mutual information
    assert entropy_2_4 > 0
    assert entropy_2_2_left >0
    assert entropy_2_2_right >0
    assert mutual_information > 0

#========================================#
#========================================#
#========================================#

#==== Keep in mind: unittest, pytest ====#

#========================================#
#========================================#
#========================================#