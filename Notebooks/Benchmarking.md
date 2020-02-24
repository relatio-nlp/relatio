```python
import json

text = "Sub-module available for the above is sent_tokenize. An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization. Imagine you need to count average words per sentence, how you will calculate? For accomplishing such a task, you need both sentence tokenization as well as words to calculate the ratio. Such output serves as an important feature for machine training as the answer would be numeric. Check the below example to learn how sentence tokenization is different from words tokenization."
with open("cong_gpo_taxation_0_54.json") as f:
    json_corpus=json.load(f)
len(json_corpus)
```




    54




```python
len(text)
```




    555




```python
text_json="".join(f"START_ID_{article[0]} {article[2]}" for article in json_corpus)
len(text_json)
```




    898549




```python
f"text_json is {len(text_json)//len(text)} bigger"
```




    'text_json is 1619 bigger'




```python
import nltk
from nltk.tokenize import sent_tokenize

print("nltk version:", nltk.__version__)
```

    nltk version: 3.4.5



```python
%timeit -o nltk.tokenize.sent_tokenize(text)
```

    169 µs ± 3.02 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)





    <TimeitResult : 169 µs ± 3.02 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)>




```python
%timeit -o nltk.tokenize.sent_tokenize(text_json)
```

    225 ms ± 11.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)





    <TimeitResult : 225 ms ± 11.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>




```python
import re

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
```


```python
%timeit -o split_into_sentences(text)
```

    73.2 µs ± 1.14 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)





    <TimeitResult : 73.2 µs ± 1.14 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)>




```python
%timeit -o split_into_sentences(text_json)
```

    90.2 ms ± 1.62 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)





    <TimeitResult : 90.2 ms ± 1.62 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)>




```python
import spacy
import en_core_web_sm

print("spacy version:", spacy.__version__)
nlp = spacy.load("en_core_web_sm")

print("model version:", nlp.meta["version"])
```

    spacy version: 2.1.9
    model version: 2.1.0


See https://github.com/explosion/spaCy/issues/453 




```python
nlp.meta
```




    {'accuracy': {'ents_f': 85.8587845242,
      'ents_p': 86.3317889027,
      'ents_r': 85.3909350025,
      'las': 89.6616629074,
      'tags_acc': 96.7783856079,
      'token_acc': 99.0697323163,
      'uas': 91.5287392082},
     'author': 'Explosion AI',
     'description': 'English multi-task CNN trained on OntoNotes. Assigns context-specific token vectors, POS tags, dependency parse and named entities.',
     'email': 'contact@explosion.ai',
     'lang': 'en',
     'license': 'MIT',
     'name': 'core_web_sm',
     'parent_package': 'spacy',
     'pipeline': ['tagger', 'parser', 'ner'],
     'sources': ['OntoNotes 5'],
     'spacy_version': '>=2.1.0',
     'speed': {'cpu': 6684.8046553827, 'gpu': None, 'nwords': 291314},
     'url': 'https://explosion.ai',
     'version': '2.1.0',
     'vectors': {'width': 0, 'vectors': 0, 'keys': 0, 'name': None}}




```python
%timeit -o doc = nlp(text,disable=['tagger', 'ner'])
```

    7.28 ms ± 444 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)





    <TimeitResult : 7.28 ms ± 444 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)>




```python
%timeit -o doc = nlp(text_json,disable=['tagger', 'ner'])
```

    10.3 s ± 210 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)





    <TimeitResult : 10.3 s ± 210 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>



## Text Preprocessing
1. lower
2. remove punctuation
3. remove numbers
4. remove superfluous white spaces



```python
import string
from string import digits
```


```python
sentences = nltk.tokenize.sent_tokenize(text)
sentences_json = nltk.tokenize.sent_tokenize(text_json)
```


```python
sentences_empty=[]
```


```python
def original_text_preprocessing(sentences):
    sentences = [sentence.lower() for sentence in sentences if bool(sentence) != False]
    sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences] # get rid of punctuation
    sentences = [sentence.translate(str.maketrans('', '', digits)) for sentence in sentences] # get rid of numbers
    sentences = [" ".join(sentence.split()) for sentence in sentences] # get rid of superfluous white spaces
    return sentences
```


```python
%timeit -o original_text_preprocessing(sentences)
```

    37.1 µs ± 149 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)





    <TimeitResult : 37.1 µs ± 149 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)>




```python
%timeit -o original_text_preprocessing(sentences_json)
```

    40.1 ms ± 508 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)





    <TimeitResult : 40.1 ms ± 508 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)>




```python
import sys
sys.path.append("../code")
from utils import preprocess
```


```python
preprocess(sentences)
```




    ['submodule available for the above is senttokenize',
     'an obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization',
     'imagine you need to count average words per sentence how you will calculate',
     'for accomplishing such a task you need both sentence tokenization as well as words to calculate the ratio',
     'such output serves as an important feature for machine training as the answer would be numeric',
     'check the below example to learn how sentence tokenization is different from words tokenization']




```python
%timeit -o preprocess(sentences)
```

    17.6 µs ± 199 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)





    <TimeitResult : 17.6 µs ± 199 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)>




```python
%timeit -o preprocess(sentences_json)
```

    29.2 ms ± 75.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)





    <TimeitResult : 29.2 ms ± 75.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)>




```python
!jupyter nbconvert --to markdown Benchmarking.ipynb
```

    [NbConvertApp] Converting notebook Benchmarking.ipynb to markdown
    [NbConvertApp] Writing 6492 bytes to Benchmarking.md

