# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 07:36:12 2021

@author: Nir
"""

import neural_ising as ni
import numpy as np
import matplotlib.pyplot as plt

start_kT = 1
end_kT = 4
num_T = 100
list_mutual_information = []
n = 2
m = 4

for kT in np.linspace(start_kT, end_kT, num=num_T, endpoint=False):
      mutual_information = ni.main(kT)
      list_mutual_information.append(mutual_information)  
plt.plot(np.linspace(start_kT, end_kT, num=num_T, endpoint=False), list_mutual_information, linewidth=5)
plt.title(f'{n}x{m} ising model',fontsize=25)
plt.ylabel('Averaged mutual information',fontsize=15)
plt.xlabel('kT', fontsize=15)