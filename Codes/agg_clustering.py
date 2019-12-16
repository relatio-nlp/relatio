
# coding: utf-8

# In[ ]:


############################
# SETUP ####################
############################

import os
os.chdir('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/results_tax_lemmaspacy')
import numpy as np
import sklearn
import pickle as pk
import json

def open_file(fname):
    with open(fname, 'rb') as pickle_file:
        content = pk.load(pickle_file)
    return content


############################
# MAKE DENDROGRAMS #########
############################

def make_dendrogram(role):
    content = open_file('%s_sif_dict' %role)
    import pandas as pd
    A = pd.DataFrame.from_dict(content, orient='index')
    import scipy.cluster.hierarchy as shc
    import matplotlib.pyplot as plt
    plt.figure()
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram - %s" %role)
    dend = shc.dendrogram(shc.linkage(A, method='ward'))
    plt.show()
    plt.savefig('dendrogram_%s.png' %role)

make_dendrogram('patients')
make_dendrogram('agents')
make_dendrogram('verbs')


############################
# MAKE CLUSTERS ############
############################

def make_clusters(role, num_clusters):
    content = open_file('%s_sif_dict' %role)
    
    # Format the dictionary in a pandas dataframe
    import pandas as pd
    A = pd.DataFrame.from_dict(content, orient='index')
    
    # Run clustering
    from sklearn.cluster import AgglomerativeClustering
    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(A)
    doc_clusters = list(ward.labels_)
    
    # Write results in a .csv file for inspection  
    with open('%s_agg_clustering_%s.csv' %(role, num_clusters), 'w+') as f:
        for i in range(num_clusters):
            within_i_cluster = [ idx for idx, clu_num in enumerate(doc_clusters) if clu_num == i ] 
            f.write('\n Looking at cluster %s \n' %i)
            for l in within_i_cluster:
                f.write(A.index[l] + ' ; ' + '\n')
                
    # Write results in a dictionary to later mine narratives
    my_dict = {}
    with open('%s_agg_clustering_%s.pickle' %(role, num_clusters), 'wb') as pickle_outfile:
        for i in range(num_clusters):
            within_i_cluster = [ idx for idx, clu_num in enumerate(doc_clusters) if clu_num == i ] 
            for l in within_i_cluster:
                phrase = A.index[l]
                my_dict['%s' %(phrase)] = i
        pk.dump(my_dict, pickle_outfile)
        
    # Write most frequent phrase as representative of a cluster
    my_dict_bis = {}
    with open('%s_agg_clustering_%s_labels.pickle' %(role, num_clusters), 'wb') as pickle_outfile:
        freq0 = open_file('../freq_%s.pickle' %role) 
        for i in range(num_clusters + 1):
            if i != num_clusters:
                within_i_cluster = [ idx for idx, clu_num in enumerate(doc_clusters) if clu_num == i ]
                freq1 = [ freq0[A.index[p]] for p in within_i_cluster ]
                max_freq1 = freq1.index(max(freq1))
                my_dict_bis['%s' %i] = A.index[within_i_cluster[max_freq1]]
            else: 
                my_dict_bis['%s' %i] = 'none'
        pk.dump(my_dict_bis, pickle_outfile)
        

make_clusters('patients', num_clusters = 200)
make_clusters('agents', num_clusters = 50)
make_clusters('verbs', num_clusters = 200)

