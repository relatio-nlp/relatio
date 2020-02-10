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
            """
```


```python
from gensim.models import KeyedVectors, Word2Vec

from utils import tokenize_into_sentences, filter_sentences, preprocess
from word_embedding import run_word2vec
from semantic_role_labeling import run_srl
```


```python
help(tokenize_into_sentences)
sentences = tokenize_into_sentences(text)
sentences
```

    Help on function tokenize_into_sentences in module utils:
    
    tokenize_into_sentences(document: str) -> List[str]
        Split a document in sentences.
        
        Args:
            document: The document
        
        Returns:
            List of sentences
    





    ['Sub-module available for the above is sent_tokenize.',
     'An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization.',
     'Imagine you need to count average words per sentence, how you will calculate?',
     'For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio.',
     'Such output serves as an important feature for machine training as the answer would be numeric.',
     'Check the below example to learn how sentence tokenization is different from words tokenization.']




```python
help(filter_sentences)
sentences = filter_sentences(sentences, max_sentence_length=350)
sentences
```

    Help on function filter_sentences in module utils:
    
    filter_sentences(sentences: List[str], max_sentence_length: int = -1) -> List[str]
        Filter list of sentences based on the number of characters length.
        
        Args:
            max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_sentence_length = -1 all sentences are kept.
        
        Returns:
            Filtered list of sentences.
        
        Examples:
            >>> filter_sentences(['This is a house'])
            ['This is a house']
            >>> filter_sentences(['This is a house'], max_sentence_length=15)
            ['This is a house']
            >>> filter_sentences(['This is a house'], max_sentence_length=14)
            []
    





    ['Sub-module available for the above is sent_tokenize.',
     'An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization.',
     'Imagine you need to count average words per sentence, how you will calculate?',
     'For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio.',
     'Such output serves as an important feature for machine training as the answer would be numeric.',
     'Check the below example to learn how sentence tokenization is different from words tokenization.']




```python
help(preprocess)
sentences_for_w2v = preprocess(sentences)
sentences_for_w2v
```

    Help on function preprocess in module utils:
    
    preprocess(sentences: List[str], remove_punctuation: bool = True, remove_digits: bool = True, remove_chars: str = '', lowercase: bool = True, strip: bool = True, remove_whitespaces: bool = True) -> List[str]
        Preprocess a list of sentences for word embedding.
        
        Args:
            sentence: list of sentences
            remove_punctuation: whether to remove string.punctuation
            remove_digits: whether to remove string.digits
            remove_chars: remove the given characters
            lowercase: whether to lower the case
            strip: whether to strip
            remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
        Returns:
            Processed list of sentences
        
        Examples:
            >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'])
            ['return the factorial of n an exact integer']
            >>> preprocess(['A1b c\n\nde \t fg\rkl\r\n m+n'])
            ['ab c de fg kl mn']
    





    ['submodule available for the above is senttokenize',
     'an obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization',
     'imagine you need to count average words per sentence how you will calculate',
     'for accomplishing such a task you need both sentence tokenization as well as words to calculate the ratio',
     'such output serves as an important feature for machine training as the answer would be numeric',
     'check the below example to learn how sentence tokenization is different from words tokenization']




```python
help(run_word2vec)
w2v_model = run_word2vec(
    sentences=sentences_for_w2v,
    model=Word2Vec(size=300, window=10, min_count=1, workers=1),
    pretrained_path="glove_2_word2vec.6B.300d.txt",
    save_path=None,
)
```

    Help on function run_word2vec in module word_embedding:
    
    run_word2vec(sentences: List[str], model: gensim.models.word2vec.Word2Vec, pretrained_path: str, save_path: Union[NoneType, str] = None) -> gensim.models.word2vec.Word2Vec
    



```python
for word, vocab_obj in w2v_model.wv.vocab.items():
    print_bool=False
    for sent in sentences_for_w2v:
        if word in sent.split():
            print_bool=True
            break
    if print_bool:
        print(word,vocab_obj.count)
```

    submodule 2
    available 2
    for 4
    the 6
    above 2
    is 4
    senttokenize 1
    an 3
    obvious 2
    question 2
    in 2
    your 2
    mind 2
    would 3
    be 3
    why 2
    sentence 5
    tokenization 6
    needed 2
    when 2
    we 2
    have 2
    option 2
    of 2
    word 2
    imagine 2
    you 4
    need 3
    to 4
    count 2
    average 2
    words 4
    per 2
    how 3
    will 2
    calculate 3
    accomplishing 2
    such 3
    a 2
    task 2
    both 2
    as 5
    well 2
    ratio 2
    output 2
    serves 2
    important 2
    feature 2
    machine 2
    training 2
    answer 2
    numeric 2
    check 2
    below 2
    example 2
    learn 2
    different 2
    from 2


## Semantic Role Labeling
### Warning
Make sure that you have a `srl-model-2018.05.25.tar.gz` available at https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz



```python
sentences
```




    ['Sub-module available for the above is sent_tokenize.',
     'An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization.',
     'Imagine you need to count average words per sentence, how you will calculate?',
     'For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio.',
     'Such output serves as an important feature for machine training as the answer would be numeric.',
     'Check the below example to learn how sentence tokenization is different from words tokenization.']




```python
help(run_srl)
srl_res = run_srl(
    sentences=sentences, predictor_path="srl-model-2018.05.25.tar.gz", save_path=None
)
```

    Help on function run_srl in module semantic_role_labeling:
    
    run_srl(sentences: List[str], predictor_path: str, save_path: Union[NoneType, str] = None) -> List[Dict[str, Any]]
    



```python
srl_res
```




    [{'verbs': [{'verb': 'is',
        'description': 'Sub - module available for [ARG1: the above] [V: is] [ARG2: sent_tokenize] .',
        'tags': ['O',
         'O',
         'O',
         'O',
         'O',
         'B-ARG1',
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
        'description': 'Imagine [ARG0: you] [V: need] [ARG1: to count average words per sentence , how you will calculate] ?',
        'tags': ['O',
         'B-ARG0',
         'B-V',
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
       '.']}]




```python
!jupyter nbconvert --to markdown Example.ipynb
```

    [NbConvertApp] Converting notebook Example.ipynb to markdown
    [NbConvertApp] Writing 6415 bytes to Example.md

