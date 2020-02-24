```python
import sys
sys.path.append("../code")
```


```python
text = """Sub-module available for the above is sent_tokenize.
            An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization. 
            Imagine you need to count average words per sentence, how you will calculate? 
            For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio. 
            Such output serves as an important feature for machine training as the answer would be numeric. 
            Check the below example to learn how sentence tokenization is different from words tokenization.
            The taxes the President announced will not lower work incentives. Evrika!
            """
```


```python
from utils import tokenize_into_sentences, filter_sentences, preprocess, UsedRoles
from word_embedding import run_word2vec, compute_embedding, USE, SIF_Word2Vec
from semantic_role_labeling import SRL, extract_roles, postprocess_roles
from clustering import Clustering
from sklearn.cluster import KMeans
from cooccurrence import build_df, subset_as_tuples, unique_counts, compute_pmi
```


```python
used_roles=UsedRoles()
used_roles['ARG2']=True
print(f"{used_roles.used}\n{used_roles.embeddable}\n{used_roles.not_embeddable}\n")
```

    ['ARGO', 'ARG1', 'ARG2', 'B-V', 'B-ARGM-MOD', 'B-ARGM-NEG']
    ['ARGO', 'ARG1', 'ARG2', 'B-V']
    ['B-ARGM-MOD', 'B-ARGM-NEG']
    



```python
srl = SRL("./srl-model-2018.05.25.tar.gz")
srl([" ".join(["What","are","you","doing"])])
```




    [{'verbs': [{'verb': 'are',
        'description': 'What [V: are] [ARG1: you doing]',
        'tags': ['O', 'B-V', 'B-ARG1', 'I-ARG1']},
       {'verb': 'doing',
        'description': '[ARG1: What] are [ARG0: you] [V: doing]',
        'tags': ['B-ARG1', 'O', 'B-ARG0', 'B-V']}],
      'words': ['What', 'are', 'you', 'doing']}]




```python
use = USE('./USE-4')
use(["What","are","you","doing"]).shape
```




    (512,)




```python
sif_w2v = SIF_Word2Vec("./nytimes_word2vec.model")
sif_w2v(["what","are","you","doing"]).shape
```




    (300,)




```python
kmeans=KMeans()
```


```python
sentences = tokenize_into_sentences(text)
sentences
```




    ['Sub-module available for the above is sent_tokenize.',
     'An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization.',
     'Imagine you need to count average words per sentence, how you will calculate?',
     'For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio.',
     'Such output serves as an important feature for machine training as the answer would be numeric.',
     'Check the below example to learn how sentence tokenization is different from words tokenization.',
     'The taxes the President announced will not lower work incentives.',
     'Evrika!']




```python
sentences = filter_sentences(sentences, max_sentence_length=350)
sentences
```




    ['Sub-module available for the above is sent_tokenize.',
     'An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization.',
     'Imagine you need to count average words per sentence, how you will calculate?',
     'For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio.',
     'Such output serves as an important feature for machine training as the answer would be numeric.',
     'Check the below example to learn how sentence tokenization is different from words tokenization.',
     'The taxes the President announced will not lower work incentives.',
     'Evrika!']




```python
srl_res = srl(sentences=sentences)
srl_res
```




    [{'verbs': [{'verb': 'is',
        'description': 'Sub - module [ARG1: available for the above] [V: is] [ARG2: sent_tokenize] .',
        'tags': ['O',
         'O',
         'O',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'B-V',
         'B-ARG2',
         'O']}],
      'words': ['Sub',
       '-',
       'module',
       'available',
       'for',
       'the',
       'above',
       'is',
       'sent_tokenize',
       '.']},
     {'verbs': [{'verb': 'would',
        'description': 'An obvious question in your mind [V: would] be why sentence tokenization is needed when we have the option of word tokenization .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-V',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O']},
       {'verb': 'be',
        'description': '[ARG1: An obvious question in your mind] [ARGM-MOD: would] [V: be] [ARG2: why sentence tokenization is needed when we have the option of word tokenization] .',
        'tags': ['B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'B-ARGM-MOD',
         'B-V',
         'B-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'O']},
       {'verb': 'is',
        'description': 'An obvious question in your mind would be why sentence tokenization [V: is] needed when we have the option of word tokenization .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-V',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O']},
       {'verb': 'needed',
        'description': 'An obvious question in your mind would be [ARGM-CAU: why] [ARG1: sentence tokenization] is [V: needed] [ARGM-TMP: when we have the option of word tokenization] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-ARGM-CAU',
         'B-ARG1',
         'I-ARG1',
         'O',
         'B-V',
         'B-ARGM-TMP',
         'I-ARGM-TMP',
         'I-ARGM-TMP',
         'I-ARGM-TMP',
         'I-ARGM-TMP',
         'I-ARGM-TMP',
         'I-ARGM-TMP',
         'I-ARGM-TMP',
         'O']},
       {'verb': 'have',
        'description': 'An obvious question in your mind would be why sentence tokenization is needed [ARGM-TMP: when] [ARG0: we] [V: have] [ARG1: the option of word tokenization] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-ARGM-TMP',
         'B-ARG0',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'O']}],
      'words': ['An',
       'obvious',
       'question',
       'in',
       'your',
       'mind',
       'would',
       'be',
       'why',
       'sentence',
       'tokenization',
       'is',
       'needed',
       'when',
       'we',
       'have',
       'the',
       'option',
       'of',
       'word',
       'tokenization',
       '.']},
     {'verbs': [{'verb': 'Imagine',
        'description': '[V: Imagine] [ARG1: you need to count average words per sentence , how you will calculate] ?',
        'tags': ['B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'O']},
       {'verb': 'need',
        'description': 'Imagine [ARG0: you] [V: need] [ARG1: to count average words per sentence] , how you will calculate ?',
        'tags': ['O',
         'B-ARG0',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O']},
       {'verb': 'count',
        'description': 'Imagine [ARG0: you] need to [V: count] [ARG1: average words per sentence] , how you will calculate ?',
        'tags': ['O',
         'B-ARG0',
         'O',
         'O',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O']},
       {'verb': 'will',
        'description': 'Imagine you need to count average words per sentence , how you [V: will] calculate ?',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-V',
         'O',
         'O']},
       {'verb': 'calculate',
        'description': 'Imagine you need to count average words per sentence , [ARGM-MNR: how] [ARG0: you] [ARGM-MOD: will] [V: calculate] ?',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-ARGM-MNR',
         'B-ARG0',
         'B-ARGM-MOD',
         'B-V',
         'O']}],
      'words': ['Imagine',
       'you',
       'need',
       'to',
       'count',
       'average',
       'words',
       'per',
       'sentence',
       ',',
       'how',
       'you',
       'will',
       'calculate',
       '?']},
     {'verbs': [{'verb': 'accomplishing',
        'description': 'For [V: accomplishing] [ARG1: such a task] , [ARG0: you] need both sentence tokenization as well as words to calculate the ratio .',
        'tags': ['O',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'O',
         'B-ARG0',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O']},
       {'verb': 'need',
        'description': '[ARGM-PRP: For accomplishing such a task] , [ARG0: you] [V: need] [ARG1: both sentence tokenization as well as words] [ARGM-PRP: to calculate the ratio] .',
        'tags': ['B-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'O',
         'B-ARG0',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'B-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'O']},
       {'verb': 'calculate',
        'description': 'For accomplishing such a task , [ARG0: you] need both sentence tokenization as well as words to [V: calculate] [ARG1: the ratio] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-ARG0',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'O']}],
      'words': ['For',
       'accomplishing',
       'such',
       'a',
       'task',
       ',',
       'you',
       'need',
       'both',
       'sentence',
       'tokenization',
       'as',
       'well',
       'as',
       'words',
       'to',
       'calculate',
       'the',
       'ratio',
       '.']},
     {'verbs': [{'verb': 'serves',
        'description': '[ARG0: Such output] [V: serves] [ARG1: as an important feature for machine training] [ARGM-CAU: as the answer would be numeric] .',
        'tags': ['B-ARG0',
         'I-ARG0',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'B-ARGM-CAU',
         'I-ARGM-CAU',
         'I-ARGM-CAU',
         'I-ARGM-CAU',
         'I-ARGM-CAU',
         'I-ARGM-CAU',
         'O']},
       {'verb': 'would',
        'description': 'Such output serves as an important feature for machine training as the answer [V: would] be numeric .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-V',
         'O',
         'O',
         'O']},
       {'verb': 'be',
        'description': 'Such output serves as an important feature for machine training as [ARG1: the answer] [ARGM-MOD: would] [V: be] [ARG2: numeric] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-ARG1',
         'I-ARG1',
         'B-ARGM-MOD',
         'B-V',
         'B-ARG2',
         'O']}],
      'words': ['Such',
       'output',
       'serves',
       'as',
       'an',
       'important',
       'feature',
       'for',
       'machine',
       'training',
       'as',
       'the',
       'answer',
       'would',
       'be',
       'numeric',
       '.']},
     {'verbs': [{'verb': 'Check',
        'description': '[V: Check] [ARG1: the below example] [ARGM-PRP: to learn how sentence tokenization is different from words tokenization] .',
        'tags': ['B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'B-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'I-ARGM-PRP',
         'O']},
       {'verb': 'learn',
        'description': 'Check the below example to [V: learn] [ARG1: how sentence tokenization is different from words tokenization] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'I-ARG1',
         'O']},
       {'verb': 'is',
        'description': 'Check the below example to learn [ARGM-MNR: how] [ARG1: sentence tokenization] [V: is] [ARG2: different from words tokenization] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'O',
         'B-ARGM-MNR',
         'B-ARG1',
         'I-ARG1',
         'B-V',
         'B-ARG2',
         'I-ARG2',
         'I-ARG2',
         'I-ARG2',
         'O']}],
      'words': ['Check',
       'the',
       'below',
       'example',
       'to',
       'learn',
       'how',
       'sentence',
       'tokenization',
       'is',
       'different',
       'from',
       'words',
       'tokenization',
       '.']},
     {'verbs': [{'verb': 'announced',
        'description': '[ARG1: The taxes] [ARG0: the President] [V: announced] will not lower work incentives .',
        'tags': ['B-ARG1',
         'I-ARG1',
         'B-ARG0',
         'I-ARG0',
         'B-V',
         'O',
         'O',
         'O',
         'O',
         'O',
         'O']},
       {'verb': 'will',
        'description': 'The taxes the President announced [V: will] not lower work incentives .',
        'tags': ['O', 'O', 'O', 'O', 'O', 'B-V', 'O', 'O', 'O', 'O', 'O']},
       {'verb': 'lower',
        'description': '[ARG0: The taxes the President announced] [ARGM-MOD: will] [ARGM-NEG: not] [V: lower] [ARG1: work incentives] .',
        'tags': ['B-ARG0',
         'I-ARG0',
         'I-ARG0',
         'I-ARG0',
         'I-ARG0',
         'B-ARGM-MOD',
         'B-ARGM-NEG',
         'B-V',
         'B-ARG1',
         'I-ARG1',
         'O']}],
      'words': ['The',
       'taxes',
       'the',
       'President',
       'announced',
       'will',
       'not',
       'lower',
       'work',
       'incentives',
       '.']},
     {'verbs': [], 'words': ['Evrika', '!']}]




```python
roles,sentence_index = extract_roles(srl_res)
sentence_index
```




    [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7]




```python
roles
```




    [{'ARG1': ['available', 'for', 'the', 'above'],
      'ARG2': ['sent_tokenize'],
      'B-V': ['is']},
     {'B-ARGM-MOD': ['would'],
      'ARG1': ['An', 'obvious', 'question', 'in', 'your', 'mind'],
      'ARG2': ['why',
       'sentence',
       'tokenization',
       'is',
       'needed',
       'when',
       'we',
       'have',
       'the',
       'option',
       'of',
       'word',
       'tokenization'],
      'B-V': ['be']},
     {'ARG1': ['sentence', 'tokenization'], 'B-V': ['needed']},
     {'ARGO': ['we'],
      'ARG1': ['the', 'option', 'of', 'word', 'tokenization'],
      'B-V': ['have']},
     {'ARG1': ['you',
       'need',
       'to',
       'count',
       'average',
       'words',
       'per',
       'sentence',
       ',',
       'how',
       'you',
       'will',
       'calculate'],
      'B-V': ['Imagine']},
     {'ARGO': ['you'],
      'ARG1': ['to', 'count', 'average', 'words', 'per', 'sentence'],
      'B-V': ['need']},
     {'ARGO': ['you'],
      'ARG1': ['average', 'words', 'per', 'sentence'],
      'B-V': ['count']},
     {'B-ARGM-MOD': ['will'], 'ARGO': ['you'], 'B-V': ['calculate']},
     {'ARGO': ['you'], 'ARG1': ['such', 'a', 'task'], 'B-V': ['accomplishing']},
     {'ARGO': ['you'],
      'ARG1': ['both', 'sentence', 'tokenization', 'as', 'well', 'as', 'words'],
      'B-V': ['need']},
     {'ARGO': ['you'], 'ARG1': ['the', 'ratio'], 'B-V': ['calculate']},
     {'ARGO': ['Such', 'output'],
      'ARG1': ['as', 'an', 'important', 'feature', 'for', 'machine', 'training'],
      'B-V': ['serves']},
     {'B-ARGM-MOD': ['would'],
      'ARG1': ['the', 'answer'],
      'ARG2': ['numeric'],
      'B-V': ['be']},
     {'ARG1': ['the', 'below', 'example'], 'B-V': ['Check']},
     {'ARG1': ['how',
       'sentence',
       'tokenization',
       'is',
       'different',
       'from',
       'words',
       'tokenization'],
      'B-V': ['learn']},
     {'ARG1': ['sentence', 'tokenization'],
      'ARG2': ['different', 'from', 'words', 'tokenization'],
      'B-V': ['is']},
     {'ARGO': ['the', 'President'],
      'ARG1': ['The', 'taxes'],
      'B-V': ['announced']},
     {'B-ARGM-MOD': ['will'],
      'ARGO': ['The', 'taxes', 'the', 'President', 'announced'],
      'ARG1': ['work', 'incentives'],
      'B-V': ['lower'],
      'B-ARGM-NEG': True},
     {}]




```python
postproc_roles = postprocess_roles(roles)
postproc_roles
```




    [{'ARG1': ['available', 'for', 'the', 'above'],
      'ARG2': ['senttokenize'],
      'B-V': ['is']},
     {'B-ARGM-MOD': ['would'],
      'ARG1': ['an', 'obvious', 'question', 'in', 'your', 'mind'],
      'ARG2': ['why',
       'sentence',
       'tokenization',
       'is',
       'needed',
       'when',
       'we',
       'have',
       'the',
       'option',
       'of',
       'word',
       'tokenization'],
      'B-V': ['be']},
     {'ARG1': ['sentence', 'tokenization'], 'B-V': ['needed']},
     {'ARGO': ['we'],
      'ARG1': ['the', 'option', 'of', 'word', 'tokenization'],
      'B-V': ['have']},
     {'ARG1': ['you',
       'need',
       'to',
       'count',
       'average',
       'word',
       'per',
       'sentence',
       'how',
       'you',
       'will',
       'calculate'],
      'B-V': ['imagine']},
     {'ARGO': ['you'],
      'ARG1': ['to', 'count', 'average', 'word', 'per', 'sentence'],
      'B-V': ['need']},
     {'ARGO': ['you'],
      'ARG1': ['average', 'word', 'per', 'sentence'],
      'B-V': ['count']},
     {'B-ARGM-MOD': ['will'], 'ARGO': ['you'], 'B-V': ['calculate']},
     {'ARGO': ['you'], 'ARG1': ['such', 'a', 'task'], 'B-V': ['accomplishing']},
     {'ARGO': ['you'],
      'ARG1': ['both', 'sentence', 'tokenization', 'a', 'well', 'a', 'word'],
      'B-V': ['need']},
     {'ARGO': ['you'], 'ARG1': ['the', 'ratio'], 'B-V': ['calculate']},
     {'ARGO': ['such', 'output'],
      'ARG1': ['a', 'an', 'important', 'feature', 'for', 'machine', 'training'],
      'B-V': ['serf']},
     {'B-ARGM-MOD': ['would'],
      'ARG1': ['the', 'answer'],
      'ARG2': ['numeric'],
      'B-V': ['be']},
     {'ARG1': ['the', 'below', 'example'], 'B-V': ['check']},
     {'ARG1': ['how',
       'sentence',
       'tokenization',
       'is',
       'different',
       'from',
       'word',
       'tokenization'],
      'B-V': ['learn']},
     {'ARG1': ['sentence', 'tokenization'],
      'ARG2': ['different', 'from', 'word', 'tokenization'],
      'B-V': ['is']},
     {'ARGO': ['the', 'president'], 'ARG1': ['the', 'tax'], 'B-V': ['announced']},
     {'B-ARGM-MOD': ['will'],
      'ARGO': ['the', 'tax', 'the', 'president', 'announced'],
      'ARG1': ['work', 'incentive'],
      'B-V': ['lower'],
      'B-ARGM-NEG': True},
     {}]




```python
sif_vectors, sif_statements_index, sif_funny_index =compute_embedding(sif_w2v,statements=postproc_roles,
                                                                      used_roles=used_roles)
```


```python
sif_statements_index
```




    {'ARGO': array([ 3,  5,  6,  7,  8,  9, 10, 11, 16, 17]),
     'ARG1': array([ 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]),
     'ARG2': array([ 1, 12, 15]),
     'B-V': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17])}




```python
{el:sif_vectors[el].shape for el in sif_vectors.keys()}
```




    {'ARGO': (10, 300), 'ARG1': (17, 300), 'ARG2': (3, 300), 'B-V': (18, 300)}




```python
sif_funny_index
```




    {'ARGO': [], 'ARG1': [], 'ARG2': [0], 'B-V': []}




```python
postproc_roles[0]["ARG2"]
```




    ['senttokenize']




```python
USE_vectors, USE_statements_index, USE_funny_index = compute_embedding(use,roles,used_roles)

```


```python
USE_statements_index
```




    {'ARGO': array([ 3,  5,  6,  7,  8,  9, 10, 11, 16, 17]),
     'ARG1': array([ 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]),
     'ARG2': array([ 0,  1, 12, 15]),
     'B-V': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17])}




```python
{el:USE_vectors[el].shape for el in USE_vectors.keys()}
```




    {'ARGO': (10, 512), 'ARG1': (17, 512), 'ARG2': (4, 512), 'B-V': (18, 512)}




```python
clustering = Clustering(cluster=kmeans,n_clusters={'ARGO':2, 'ARG1': 2, 'ARG2':2, 'B-V':1},
                         used_roles=used_roles)
```


```python
clustering.fit(vectors=sif_vectors,sample_size=None)
```


```python
{el:clustering._cluster[el].labels_ for el in clustering._cluster.keys()}
```




    {'ARGO': array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int32),
     'ARG1': array([1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=int32),
     'ARG2': array([1, 0, 1], dtype=int32),
     'B-V': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)}




```python
clustering_res = clustering.predict(vectors=sif_vectors)
clustering_res
```




    {'ARGO': array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int32),
     'ARG1': array([1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=int32),
     'ARG2': array([1, 0, 1], dtype=int32),
     'B-V': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)}




```python
df = build_df(postproc_roles,clustering_res,sif_statements_index,used_roles)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ARGO</th>
      <th>ARG1</th>
      <th>ARG2</th>
      <th>B-V</th>
      <th>B-ARGM-MOD</th>
      <th>B-ARGM-NEG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>would</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>will</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>12</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>would</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>13</th>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>14</th>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>15</th>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>will</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
subset_tuple = subset_as_tuples(df,used_roles)
subset_tuple = subset_as_tuples(df,used_roles,{"ARGO","ARG1","B-V","B-ARGM-MOD","B-ARGM-NEG"})
subset_tuple = subset_as_tuples(df,used_roles,{"ARGO","ARG1","B-V"})

subset_tuple
```




    [(0, 0, 0),
     (0, 1, 0),
     (0, 1, 0),
     (0, 1, 0),
     (0, 0, 0),
     (0, 1, 0),
     (1, 1, 0),
     (1, 1, 0),
     (1, 1, 0)]




```python
unique_counts(subset_tuple)
```




    {(0, 0, 0): 2, (0, 1, 0): 4, (1, 1, 0): 3}




```python
results_dic = compute_pmi(subset_tuple)
results_dic
```




    {(0, 0, 0): -3.9889840465642745,
     (0, 1, 0): -4.548599834499697,
     (1, 1, 0): -4.143134726391533}




```python
!jupyter nbconvert --to markdown Example.ipynb
```

    [NbConvertApp] Converting notebook New_example.ipynb to markdown
    [NbConvertApp] Writing 27872 bytes to New_example.md

