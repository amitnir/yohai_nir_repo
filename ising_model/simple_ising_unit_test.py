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

lattices = np.array(si.f_ising_creator(n, m))
print(f'The input is: n = {n}, m = {m}, kT = {kT}')
print(f'The shape of the lattices is: {lattices.shape}')


left_lattices, right_lattices = si.f_splitter(lattices) # split all lattices into left and right ones
print(f'The shape of left lattices is: {left_lattices.shape}\nThe shape of right lattices is: {right_lattices.shape}')
entropy_2_4, energy_full = si.f_entropy(lattices, J, kT) # calculate the entropy of the big lattices
print('The entropy of the big lattices is: %.2f' %entropy_2_4)
entropy_2_2_left, energy_left = si.f_entropy(left_lattices, J, kT) # calculate the entropy of the small left lattices
print('The entropy of the left lattices is: %.2f' %entropy_2_2_left)
entropy_2_2_right, energy_right = si.f_entropy(right_lattices, J, kT) # calculate the entropy of the small right lattices
print('The entropy of the right lattices is: %.2f' %entropy_2_2_right)
mutual_information = si.f_mutual_informations(entropy_2_4, entropy_2_2_right, entropy_2_2_left) # calculate the mutual information
print('The mutual information is: %.2f' %mutual_information)
print('\n==== Examples: ====\n')

print(f'Lattice number {lat_num}:\n')
print(f'full lattice :\n{lattices[lat_num]}')
print(f'left lattice :\n{left_lattices[lat_num]}')
print(f'right lattice :\n{right_lattices[lat_num]}')
print(f'energy full :\n{energy_full[lat_num]}')
print(f'energy left :\n{energy_left[lat_num]}')
print(f'energy right :\n{energy_right[lat_num]}')