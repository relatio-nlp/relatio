# Sentence Embeddings
# (Based on Arora et al. 2017)

# Load SIF dict directly in case it has already been created
# import pickle as pk
# with open('nyt_sif_dict.p', 'rb') as pickle_file:
#     sif_dict = pk.load(pickle_file)

# Load pre-trained model
# from gensim.models import KeyedVectors, Word2Vec
# pretrained_path = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/word_embeddings/glove_2_word2vec.6B.300d.txt'
# model = KeyedVectors.load_word2vec_format(pretrained_path, binary = False)

# Import libraries
import os
import pickle as pk
import numpy as np
# Set directory
os.chdir('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/')

# Load re-trained model (trained on entire NYT corpus)
from gensim.models import KeyedVectors, Word2Vec
retrained_path = 'nytimes_word2vec.model'
# retrained_path = 'nytimes_word2vec_V2.model' # Model ONLY trained on NYT
model = Word2Vec.load(retrained_path)

print('Model loaded.')

# Create a word count dictionary based on the trained model
word_count_dict = {}
for word, vocab_obj in model.wv.vocab.items():
    word_count_dict[word] = vocab_obj.count

# Create a dictionary that maps from a word to its frequency and then use the frequency of
# a word to compute its sif-weight (saved in sif_dict)
sif_dict = {}
for word, count in word_count_dict.items():
    sif_dict[word] = .001 / (.001 + count)
# pk.dump(sif_dict, open('sif_dict.p', 'wb'))

print('SIF dictionary built!')

from sklearn import preprocessing
# Get the SIF vectors
def get_sif_vec_dict(Itemlist):
    itemlist = Itemlist.copy()
    sif_vec_dict = {}
    not_found = []
    print('Number of items: {0}'.format(len(itemlist)))
    for tokens in itemlist:
        try:
            sif_vec = np.mean([sif_dict[one_token] * model.wv[one_token] for one_token in tokens], axis=0)
            # Normalize the vector
            sif_vec = preprocessing.normalize(sif_vec.reshape(1, -1), norm="l2")
            sif_vec_dict["{0}".format('_'.join(tokens))] = list(sif_vec[0])
        except:
            not_found.append(tokens)
    not_found_unique = [list(item) for item in set(tuple(row) for row in not_found)]
    print('Unique items not found in embeddings: {0}'.format(len(not_found_unique)))
    print('Examples of tokens not found: {0}'.format(not_found_unique[0:10]))
    sif_vec_dict_flatten = [s for s in list(sif_vec_dict.values())]
    print('Unique items found: {0}'.format(len(sif_vec_dict_flatten)))
    return sif_vec_dict, sif_vec_dict_flatten


# Perform multidimensional scaling
# NB: in the end, we chose not do it.
# from sklearn.manifold import MDS
def multidim_scaling(vector_list):
    a = np.array(vector_list).astype(np.float32)
    # embedding = MDS(n_components=100)
    # a_mds = embedding.fit_transform(a)
    # return a_mds
    return a

# Set up PCA
# PCA: get first principal component
from sklearn.decomposition import PCA
pca = PCA(n_components=1)

def do_pca(vector_list):
    vectorM = np.transpose(np.asarray(list(vector_list)))
    pca.fit(vectorM)
    pc = pca.components_
    vectorM_proj = vectorM.dot(pc.transpose()) * pc
    vectorM_norm = vectorM - vectorM_proj
    print('The shape of the array is: {0}'.format(vectorM_norm.shape))
    return vectorM_norm

print('Functions defined, now fetching files.')

# Get list of unique agents, patients, and verbs from processed SRL output
path_to_dicts = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_processed_tax_lemmaspacy'
allFiles = os.listdir(path_to_dicts)
print('There are {0} files in total.'.format(len(allFiles)))

# Define function to extract agent, patients, and verbs from dictionary
def get_narr_from_dict(d_):
    d_agent = d_['agent'].split('_')
    d_patient = d_['patient'].split('_')
    d_verb = d_['verb'].split('_')
    return(d_agent, d_patient, d_verb)

agents_raw = []
patients_raw = []
verbs_raw = []
counter = 0
for f in allFiles:
    try:
        counter += 1
        if counter % 1000 == 0:
            print('At file no. {0}!'.format(counter))
            dict_lst = eval(open('srl_output_processed_tax_lemmaspacy/' + f, 'r').read())
            for d in dict_lst:
                dic_agents, dic_patients, dic_verbs = get_narr_from_dict(d)
                agents_raw.append(dic_agents)
                patients_raw.append(dic_patients)
                verbs_raw.append(dic_verbs)
    except:
        print('Did not work:', f)

print('Number of agents: {0}'.format(len(agents_raw)))
print('Number of patients: {0}'.format(len(patients_raw)))
print('Number of verbs: {0}'.format(len(verbs_raw)))

# Drop duplicate agents, patients, and verbs

agents = [list(item) for item in set(tuple(row) for row in agents_raw)]
print('Number of unique agents: {0}'.format(len(agents)))
patients = [list(item) for item in set(tuple(row) for row in patients_raw)]
print('Number of unique patients: {0}'.format(len(patients)))
verbs = [list(item) for item in set(tuple(row) for row in verbs_raw)]
print('Number of unique verbs: {0}'.format(len(verbs)))

# Filter for long agents/ patients

len_a = [len(a) for a in agents_raw]
len_p = [len(p) for p in patients_raw]
len_v = [len(v) for v in verbs_raw]

import numpy as np
from matplotlib import pyplot as plt

# Create histogram and save
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(len_a, bins=20)
axs[1].hist(len_p, bins=20)
plt.savefig('len_distr_agents_patients')

a95 = np.percentile(len_a, 95)
p95 = np.percentile(len_p, 95)
print('Threshold for dropping long agents:', a95)
print('Threshold for dropping long patients:', p95)

agents = [a for a in agents if len(a) < a95]
patients = [p for p in patients if len(p) < p95]

print('Narratives extracted. Now calculating embeddings.')

# Calculate embeddings for agents/ patients/ verbs, perform MDS, and save outputs

print('Dealing with agents')
agents_dic, agents_lst = get_sif_vec_dict(agents)
print('Agents dicts calculated.')
vectors_mds_agents = multidim_scaling(agents_lst)
print('MDS on agents done.')
with open('results_tax_lemmaspacy/agents_sif_dict', 'wb') as outfile:
    pk.dump(agents_dic, outfile)
with open('results_tax_lemmaspacy/agents_sif_list_mds', 'wb') as outfile:
    pk.dump(vectors_mds_agents, outfile)

print('Dealing with patients')
patients_dic, patients_lst = get_sif_vec_dict(patients)
print('Patient dicts calculated.')
vectors_mds_patients = multidim_scaling(patients_lst)
print('MDS on patients done.')
with open('results_tax_lemmaspacy/patients_sif_dict', 'wb') as outfile:
    pk.dump(patients_dic, outfile)
with open('results_tax_lemmaspacy/patients_sif_list_mds', 'wb') as outfile:
    pk.dump(vectors_mds_patients, outfile)

print('Dealing with verbs')
verbs_dic, verbs_lst = get_sif_vec_dict(verbs)
print('Verb dicts calculated.')
vectors_mds_verbs = multidim_scaling(verbs_lst)
print('MDS on verbs done.')
with open('results_tax_lemmaspacy/verbs_sif_dict', 'wb') as outfile:
    pk.dump(verbs_dic, outfile)
with open('results_tax_lemmaspacy/verbs_sif_list_mds', 'wb') as outfile:
    pk.dump(vectors_mds_verbs, outfile)


print('Job completed successfully.')
