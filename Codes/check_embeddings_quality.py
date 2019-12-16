
# coding: utf-8

# In[ ]:


import os 
os.chdir('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/results_tax')

import pickle as pk
def open_file(fname):
    with open(fname, 'rb') as pickle_file:
        content = pk.load(pickle_file)
    return content

a_dic = open_file('patients_sif_dict')

mylistofagents = []
for key, value in a_dic.items() :
    mylistofagents.append(key)

import pandas as pd 
df = pd.DataFrame.from_dict(a_dic, orient='index') 

vec_num = 200 # change this number here to look at various vectors

print('Looking at vector %s' %df.index[vec_num])

n_nearest = 10

import numpy as np
ref_vec = np.array(a_dic[df.index[vec_num]]).reshape(1,-1)

distance_matrix = pairwise_distances(ref_vec, df).reshape(-1,) # distances to the center

n_closest = list(np.argpartition(distance_matrix, n_nearest)[:n_nearest]) # find the n closest documents

print('The %s nearest vectors are:' %n_nearest)

for l in n_closest: 
    print(df.index[l])

