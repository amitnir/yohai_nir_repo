def f_ising_creator(n, m):
  """create a list of all matrix combinations"""

  x = product([1, -1], repeat=n*m)
  x = np.reshape(list(x), (-1, n, m))

  for i in x:
    i[i==0] = -1

  lattice_combinations = x

  return lattice_combinations


def f_splitter(lattice):
  """split our lattices into left and right ones"""

  left_lattice, right_lattice = np.split(lattice, 2, axis=1)
  return left_lattice, right_lattice


def f_energy_lattice(my_lattice, n, m):
  """calculate energy lattice of nxm lattice"""

  energy_lattice = np.zeros((n,m))

  r_end = my_lattice.shape[0]-1
  c_end = my_lattice.shape[1]-1
  for i in range(n):
    for j in range(m):
      # 4 corners:
      if i == 0 and j == 0:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[0,c_end] + my_lattice[0,1] + my_lattice[1,0] + my_lattice[r_end,0]) 
      elif i == r_end and j == 0:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[r_end-1,0] + my_lattice[0,0] + my_lattice[r_end,1] + my_lattice[r_end,c_end]) 
      elif i == 0 and j == c_end:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[0,c_end-1] + my_lattice[0,0] + my_lattice[1,c_end] + my_lattice[r_end,c_end]) 
      elif i == r_end and j == c_end:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[0,0] + my_lattice[r_end,c_end-1] + my_lattice[r_end,0] + my_lattice[0,c_end]) 
      
      # first and last row:
      elif i == 0:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[0,j-1] + my_lattice[0,j+1] + my_lattice[1,j] + my_lattice[r_end,j])
        
      elif i == r_end:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[r_end,j-1] + my_lattice[r_end,j+1] + my_lattice[r_end-1,j] + my_lattice[0,j])
       
      # first and last column:
      elif j == 0:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[i,j+1] + my_lattice[i-1,j] + my_lattice[i+1,j] + my_lattice[i,c_end])
      elif j == c_end:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[i,0] + my_lattice[i,j-1] + my_lattice[i-1,j] + my_lattice[i+1,j])
      
      # regular ones:  
      else:
        energy_lattice[i,j] = J * my_lattice[i,j] * (my_lattice[i,j-1] + my_lattice[i,j+1] + my_lattice[i+1,j] + my_lattice[i-1,j]) 
    
  return energy_lattice

  
def f_entropy(energy_lattice):
  """calculate entropy of nxm matrix"""

  matrix = energy_lattice
  entropy = -sum(sum(np.multiply(np.exp(-matrix/kT)/(sum(sum(np.exp(-matrix/kT)))),np.log(np.exp(-matrix/kT)/(sum(sum(np.exp(-matrix/kT))))))))

  return entropy


def f_mutual_informations(S_ab, S_a, S_b):
  """calculate mutual information"""
  
  mutual_informations = (S_a + S_b) - S_ab

  return mutual_informations


def average_mutual(mutual_informations, energies_2_4):
  """calculate the averaged mutual information"""

  energy_list = []
  for i in energies_2_4:
    energy_list.append(sum(sum(i)))

  Z = sum(np.exp(-(np.array(energy_list))/kT))
  probabilities = np.exp(-np.array(energy_list)/kT)
  expected_value = (1/Z) * sum(np.multiply(mutual_informations,  probabilities))
  return expected_value 
