# Semantic Role Labeling on NYT Corpus: Process the Outputs
# In this script, we process the SRL outputs.
# Note that this file is named "simple" because we only extract agents, patients
# and verbs in clear cases (e.g., we suppress modal verbs).
# We load and process the SRL output file by file.

# This script does not require multiprocessing (reasonably fast).

# Set up text cleaning functions
print('File version: 13 October')

import nltk

# Define contractions to be expanded
CONTRACTION_MAP = {"ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he'll've": 'he he will have', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'd've": 'I would have', "I'll": 'I will', "I'll've": 'I will have', "I'm": 'I am', "I've": 'I have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there would', "there'd've": 'there would have', "there's": 'there is', "they'd": 'they would', "they'd've": 'they would have', "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what's": 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will', "you'll've": 'you will have', "you're": 'you are', "you've": 'you have'}

def expand_contractions(text_str, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text_str)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Remove all digits
from string import digits
translator = str.maketrans('', '', digits)

# Get stop words from NLTK and extend
from string import ascii_lowercase
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['nevertheless', 'would', 'nether', 'the', 'in', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five',
         'six', 'seven', 'eight', 'nine', 'ten', 'quot', 'across', 'among', 'beside', 'however', 'yet',
         'within', 'et'] + list(ascii_lowercase))

# Get SnowballStemmer from NLTK
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english', ignore_stopwords=True)

# Use NLTK tokenizer to remove all non-alphanumeric tokens
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

# Combine all cleaning steps in one function
def normalize_doc(text_str):
    # Extend contractions ("you've" to "you have", etc.)
    text_str = expand_contractions(text_str)
    # Set all text to lower
    text_str = text_str.lower()
    # Remove digits
    text_str = text_str.translate(translator)
    # Tokenize (this will also eliminate all special chars and punctuation)
    text_str_lst = tokenizer.tokenize(text_str)
    # Remove stop words
    text_str_lst = [tok for tok in text_str_lst if tok not in stop_words]
    # Do stemming
    # text_str_lst = [stemmer.stem(tok) for tok in text_str_lst]
    # Do lemmatization
    text_str_lst = [lemmatizer.lemmatize(tok) for tok in text_str_lst]
    text_str = ' '.join(text_str_lst)
    return text_str

import os, json, re, itertools

# Open a SRL output file and extract the list of dictionaries
def get_dict_list(f_):
    dict_lst = eval(open(f_, 'r').read())
    return dict_lst

def clean_sem_role_lst(lst):
    # lst = list(itertools.chain.from_iterable(lst))
    lst = [normalize_doc(tok) for tok in lst]
    lst = list(filter(('').__ne__, lst))
    lst = list(filter((' ').__ne__, lst))
    if lst:
        lst_str = '_'.join(lst)
    else:
        lst_str = ''
    return lst_str

# Clean the SRL output
def clean_srl_output(dict_, dict_idx):
    tag_masterlist = []
    narrative_dictList = []
    for d_ in dict_['verbs']:
        lst_ = d_['tags']
        tag_masterlist.append(lst_)
    for n in range(0, len(tag_masterlist)):
        tags = tag_masterlist[n]
        idx_lst = []
        for t in tags:
            if bool(re.search('B-ARG[0-9]', t)):
                idx = int(t[-1:])
                idx_lst.append(idx)
        if len(idx_lst) > 0:
            ag_idx = min(idx_lst)
            pat_idx = max(idx_lst)
            if  ag_idx != pat_idx:
                if (pat_idx - ag_idx) > 1:
                    pat_idx = ag_idx + 1
                # Get agents
                indices_agent = [i for i, token in enumerate(tags) if token == 'B-ARG{0}'.format(ag_idx) or token == 'I-ARG{0}'.format(ag_idx)]
                agent_raw = [token.lower() for m, token in enumerate(dict_['words']) if m in indices_agent]
                agent = clean_sem_role_lst(agent_raw)
                # Get patients
                indices_patient = [i for i, token in enumerate(tags) if token == 'B-ARG{0}'.format(pat_idx) or token == 'I-ARG{0}'.format(pat_idx)]
                patient_raw = [token.lower() for m, token in enumerate(dict_['words']) if m in indices_patient]
                patient = clean_sem_role_lst(patient_raw)
                # Get verb
                indices_verb = [i for i, token in enumerate(tags) if token == 'B-V']
                verb_raw = [token.lower() for m, token in enumerate(dict_['words']) if m in indices_verb]
                verb = clean_sem_role_lst(verb_raw)
                # Write narrative for given verb to a dictionary
                verbDict = {'dict_id' : dict_idx, 'verb_dict_id' : n, 'agent' : agent, 'patient' : patient, 'verb' : verb}
                narrative_dictList.append(verbDict)
    return narrative_dictList

# Fetch all files from the srl_output folder
saveDir1 = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_tax/'
os.chdir(saveDir1)
allFiles = os.listdir(saveDir1)
print('There are {0} files in total.'.format(len(allFiles)))

# Check if file has already been processed (i.e., has already been saved in the srl_output_processed folder)
saveDir2 = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_processed_tax/'
filesToHandle = []
for f in allFiles:
    if not os.path.isfile(saveDir2 + f):
        filesToHandle.append(f)
del f
print('There are {0} files left to do.'.format(len(filesToHandle)))

def handle_one_file(f):
    try:
        narrative_masterlist = []
        dict_list = get_dict_list(f)
        for Dict_Idx, Dict in enumerate(dict_list):
            narrative_list = clean_srl_output(Dict, Dict_Idx)
            narrative_masterlist.extend(narrative_list)
        # Save narratives
        with open(saveDir2 + f, 'w+') as outfile:
            json.dump(narrative_masterlist, outfile)
    except:
        print('Did not work: {0}'.format(f))

# Define function to deal with one file
def handle_one_file(f):
    narrative_masterlist = []
    dict_list = get_dict_list(f)
    for Dict_Idx, Dict in enumerate(dict_list):
        narrative_list = clean_srl_output(Dict, Dict_Idx)
        narrative_masterlist.extend(narrative_list)
        # Save narratives
    with open(saveDir2 + f, 'w+') as outfile:
            json.dump(narrative_masterlist, outfile)

# Iterate over all files
from joblib import Parallel, delayed
import multiprocessing
n_cores = multiprocessing.cpu_count()
Parallel(n_jobs=n_cores)(delayed(handle_one_file)(file)
                         for file in filesToHandle)
