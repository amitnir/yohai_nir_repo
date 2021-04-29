# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 07:36:12 2021

@author: Nir
"""

import neural_ising as ni
import numpy as np
import matplotlib.pyplot as plt

start_kT = 0.01
end_kT = 30
num_T = 30
n = 2
m = 4

list_mutual_information = [
  ni.calcualte_entropy_for_one_temperature(kT) for kT in np.linspace(start_kT, end_kT, num=num_T, endpoint=False)
]
plt.plot(np.linspace(start_kT, end_kT, num=num_T, endpoint=False), list_mutual_information, linewidth=5)
plt.title(f'{n}x{m} ising model',fontsize=25)
plt.ylabel('Averaged mutual information',fontsize=15)
plt.xlabel('kT', fontsize=15)