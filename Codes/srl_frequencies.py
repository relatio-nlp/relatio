
# coding: utf-8

# In[ ]:


# Import libraries
import os
import json

# Set directory
my_dir = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_processed_tax_lemmaspacy'
os.chdir(my_dir)

# Look for processed srl files
import glob
filelist = glob.glob(my_dir + '/*')

# Compute frequencies for each semantic role 

freq_agent = {}
freq_verb = {}
freq_patient = {}

for file in filelist: 
    with open(file, 'rb') as f:
        srl_json = json.load(f)
        
    for d in srl_json:
        if freq_agent.get(d['agent']):
            freq_agent[d['agent']] = freq_agent[d['agent']] + 1
        else: 
            freq_agent[d['agent']] = 1
            
    for d in srl_json:
        if freq_verb.get(d['verb']):
            freq_verb[d['verb']] = freq_verb[d['verb']] + 1
        else: 
            freq_verb[d['verb']] = 1         
            
    for d in srl_json:
        if freq_patient.get(d['patient']):
            freq_patient[d['patient']] = freq_patient[d['patient']] + 1
        else: 
            freq_patient[d['patient']] = 1
            
import pickle as pk 

with open('../freq_agents.pickle', 'wb') as outfile:
    pk.dump(freq_agent, outfile)
with open('../freq_verbs.pickle', 'wb') as outfile:
    pk.dump(freq_verb, outfile)   
with open('../freq_patients.pickle', 'wb') as outfile:
    pk.dump(freq_patient, outfile)            
            
        
# Sort semantic role phrases per frequency
sorted(freq_agent.items(), key=lambda x:x[1], reverse=True)  
sorted(freq_verb.items(), key=lambda x:x[1], reverse=True)  
sorted(freq_patient.items(), key=lambda x:x[1], reverse=True)  

