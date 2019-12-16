# Import libraries
import re 
import json

# Functions 
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def remove_brackets(test_str): 
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret

def load_data_from_nytimes(year):
    import psycopg2
    import logging
    logging.basicConfig(format='%(asctime)s  %(message)s',
        datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)
    import sys
    sys.path.append("/cluster/work/lawecon/Work/goessmann/python_common/")
    import database_connection

    # connect to database
    logging.info('opening db connection')
    con, cur = database_connection.connect()

    # define sql query as string
    sql = """
    SELECT
        title,
        body
    FROM
        corp_nytimes
    WHERE
        publication_date >= '%s-01-01' and publication_date < '%s-01-01'
        """ % (year, year + 1)

    # execute query
    cur.execute(sql)

    # obtain all result rows from the database server
    rows = cur.fetchall()

    logging.info('fetched all rows from db into memory, closing db connection')
    con.close()
    return rows

import string
from string import digits

with open('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/unigram_sentences_nytimes.txt', 'w') as f:
    for year in range(1987,2008):
        articles = load_data_from_nytimes(year)
        for article in articles:
            title, body = article
            body = body.replace('LEAD:', '')
            sentences = split_into_sentences(body)
            sentences = list(dict.fromkeys(sentences))
            sentences = [title] + sentences
            sentences = [sentence.lower() for sentence in sentences if bool(sentence) != False]
            sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences] # get rid of punctuation
            sentences = [sentence.translate(str.maketrans('', '', digits)) for sentence in sentences] # get rid of numbers
            sentences = [" ".join(sentence.split()) for sentence in sentences] # get rid of superfluous white spaces
            for sentence in sentences:
                f.write(sentence + '\n')


