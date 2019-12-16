
# coding: utf-8

# In[ ]:


# Import libraries
import os
import json

# Set directory
my_dir = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_processed_tax_lemmaspacy'
os.chdir(my_dir)

# Open pickle file
import pickle as pk
def open_file(fname):
    with open(fname, 'rb') as pickle_file:
        content = pk.load(pickle_file)
    return content

# Look for processed srl files
import glob
filelist = glob.glob(my_dir + '/*')

# Load cluster dictionaries
patient_clusters = open_file('../results_tax_lemmaspacy/patients_agg_clustering_200.pickle') #results_tax_no_lemma
agent_clusters = open_file('../results_tax_lemmaspacy/agents_agg_clustering_50.pickle')
verb_clusters = open_file('../results_tax_lemmaspacy/verbs_agg_clustering_50.pickle')

# Load representative cluster phrases

P = open_file('../results_tax_lemmaspacy/patients_agg_clustering_200_labels.pickle')
A = open_file('../results_tax_lemmaspacy/agents_agg_clustering_50_labels.pickle')
V = open_file('../results_tax_lemmaspacy/verbs_agg_clustering_200_labels.pickle')

num_agent_clu = len(A) - 1
num_patient_clu = len(P) - 1
num_verb_clu = len(V) - 1


# In[ ]:


# Assign to each semantic role its cluster

json_clusters = []
for file in filelist:
    with open(file, 'rb') as f:
        srl_json = json.load(f)
    # Assign to each semantic role a cluster
    for d in srl_json:
        #create dict to store results 
        my_dict = {}
        # keep the same indices as Philine
        my_dict['dict_id'] = d['dict_id']
        my_dict['verb_dict_id'] = d['verb_dict_id']
        
        # find the relevant cluster for each semantic role
        num_cluster = agent_clusters.get(d['agent'])
        if num_cluster: # To handle KeyErrors.
            my_dict['agent'] = num_cluster
        else: 
            my_dict['agent'] = num_agent_clu
            
        num_cluster = patient_clusters.get(d['patient'])
        if num_cluster:
            my_dict['patient'] = num_cluster
        else: 
            my_dict['patient'] = num_patient_clu
            
        num_cluster = verb_clusters.get(d['verb'])
        if num_cluster: 
            my_dict['verb'] = num_cluster
        else: 
            my_dict['verb'] = num_verb_clu
            
        # append to my json
        json_clusters.append(my_dict)
        
# Count frequencies
import numpy as np
    
# Only keep trios 
M = np.zeros((num_agent_clu, num_verb_clu, num_patient_clu))
for d in json_clusters:
    if d['agent'] != num_agent_clu:
        if d['verb'] !=num_verb_clu:
            if d['patient'] != num_patient_clu:
                    M[d['agent'],d['verb'],d['patient']] = M[d['agent'],d['verb'],d['patient']] + 1

total_counts = np.sum(M)                    
                    
# Top N narratives (totally unsupervised)
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

N = 100
top_idx = np.transpose(np.array(largest_indices(M, N)))

# with open('../narrative_examples.csv', 'w+') as f:
for i in range(N):
    agent_idx = top_idx[i][0]
    verb_idx = top_idx[i][1]
    patient_idx = top_idx[i][2]
    percentage = M[agent_idx, verb_idx, patient_idx]*100/total_counts
    print('Narrative %s:' %i)
    print('%s - %s - %s' %(A[str(agent_idx)],V[str(verb_idx)],P[str(patient_idx)]))
    print('This narrative occurs %s percent of the time. \n' %percentage)


# In[ ]:


# Top N narratives for a specific agent cluster

N = 20

A #list agent clusters: 24, 30, 25, 31
num_cluster_of_interest = 25

MA = M[:,num_cluster_of_interest,:]

total_counts = np.sum(MA) 

top_idx = np.transpose(np.array(largest_indices(MA, N)))

# with open('../narrative_examples.csv', 'w+') as f:
for i in range(N):
    verb_idx = top_idx[i][0]
    patient_idx = top_idx[i][1]
    percentage = MA[verb_idx, patient_idx]*100/total_counts
    print('Narrative %s:' %i)
    print('%s - %s - %s' %(A[str(num_cluster_of_interest)],V[str(verb_idx)],P[str(patient_idx)]))
    print('This narrative occurs %s percent of the time. \n' %percentage)


# In[ ]:


# Looking at specific verbs 

verb_of_interest = 'provide' # increase, decrease, reduce, raise, require, jeopardize, provide, hire

json_clusters = []
for file in filelist:
    with open(file, 'rb') as f:
        srl_json = json.load(f)
        
    # Assign to each semantic role a cluster
    for d in srl_json:
        
        if d['verb'] == verb_of_interest: # verb I want to look at
            
            my_dict = {}

            # keep the same indices as Philine
            my_dict['dict_id'] = d['dict_id']
            my_dict['verb_dict_id'] = d['verb_dict_id']

            # find the relevant cluster for each semantic role
            num_cluster = agent_clusters.get(d['agent'])
            if num_cluster: # To handle KeyErrors.
                my_dict['agent'] = num_cluster
            else: 
                my_dict['agent'] = num_agent_clu

            num_cluster = patient_clusters.get(d['patient'])
            if num_cluster:
                my_dict['patient'] = num_cluster
            else: 
                my_dict['patient'] = num_patient_clu

            # append to my json
            json_clusters.append(my_dict)
        
# Count frequencies
import numpy as np
    
# Only keep trios 
M = np.zeros((num_agent_clu, num_patient_clu))
for d in json_clusters:
    if d['agent'] != num_agent_clu:
            if d['patient'] != num_patient_clu:
                    M[d['agent'],d['patient']] = M[d['agent'],d['patient']] + 1

total_counts = np.sum(M)                      
                    
# Top N values
def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

N = 20
top_idx = np.transpose(np.array(largest_indices(M, N)))

for i in range(N):
    agent_idx = top_idx[i][0]
    patient_idx = top_idx[i][1]
    percentage = M[agent_idx,patient_idx]*100/total_counts
    print('Narrative %s:' %i)
    print('%s - %s - %s \n' %(A[str(agent_idx)],verb_of_interest,P[str(patient_idx)])) 
    print('This narrative represents %s percent. \n' %percentage)

