# Semantic Role Labeling on NYT Corpus
# In this notebook, we run semantic role labeling on NYT articles related to taxation/fiscal policy.
# WE load and process the data on a yearly basis, from 1987 to 2007.
# Germain Gauthier and Philine Widmer (30/08/2019)

# Import libraries
import time
import allennlp
import re, os
import json

print('File version 26 September 2019 -- 1')

# os.chdir('/cluster/home/widmerph/nlp2/nyt')
os.chdir('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/')

# Load pre-trained SRL model
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path('srl_model/srl-model-2018.05.25.tar.gz')
print('Pre-trained SRL model successfully loaded.')

def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    # websites2 = "[.](com|net|org|io|gov)[.]" #end of sentence
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

def load_data_from_nyt(date_):
    import logging, sys
    logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)
    sys.path.append("/cluster/work/lawecon/Work/goessmann/python_common/")
    import database_connection
    logging.info('Opening db connection')
    con, cur = database_connection.connect()
    # Define sql query as string
    sql = """
    SELECT
        id,
        title,
        body,
        publication_date,
        online_sections
    FROM
        corp_nytimes
    WHERE
        publication_date = '{0}' AND
        ((lower(body) like '%% tax %%') OR
         (lower(body) like '%% fiscal %%') OR
         (lower(body) like '%% taxation %%') OR
         (lower(body) like '%% taxation %%') OR
         (lower(body) like '%% taxed %%') OR
         (lower(body) like '%% taxable %%') OR
         (lower(body) like '%% taxman %%') OR
         (lower(body) like '%% taxpayer %%') OR
         (lower(body) like '%% taxing %%') OR
         (lower(body) like '%% taxing %%'))
    """.format(date_)
    # Execute query
    cur.execute(sql)
    # Obtain all result rows from the database server
    rows = cur.fetchall()
    logging.info('Fetched all rows from db into memory, closing db connection')
    con.close()
    # The object "rows" is of type list (of strings); e.g., rows[0][2] gives body of first article
    return rows

import spacy
nlp = spacy.load("en_core_web_sm")
def sent_splitter(str_):
    sents_lst = [i for i in nlp(str_).sents]
    return sents_lst

import datetime
def getBetweenDay(begin_date, end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list

dates_all_raw = getBetweenDay('1987-01-01', '2007-06-19')
print('Number of all days:', len(dates_all_raw))

# check_file_ = 'check_file_tax'
# done_lst = []
# with open(check_file_, 'r') as check_:
#     for row_ in check_:
#         done_lst.append(row_.strip('\n'))
#
# dates_all = []
# for d_raw in dates_all_raw:
#     if d_raw not in done_lst:
#         dates_all.append(d_raw)

dates_all = dates_all_raw.copy()
print('Number of dates to do:', len(dates_all))

savePath = '/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_tax_2sents/'

max_sentence = 350
def process_date(date):
    Rows = load_data_from_nyt(date)
    for R in Rows:
        id, title, body, publication_date, online_sections = R
        FileName = savePath + 'srl_{0}_{1}'.format(publication_date, id)
        if os.path.isfile(FileName):
            pass
        else:
            Sents_list = []
            Sents_list_raw = split_into_sentences(body)
            for i in range(0, len(Sents_list_raw)):
                if ('tax' in str(Sents_list_raw[i])) or ('fiscal' in str(Sents_list_raw[i])):
                    Sents_list.append(str(Sents_list_raw[i]))
                    try:
                        Sents_list.append(str(Sents_list_raw[i+1]))
                    except:
                        pass
                    try:
                        Sents_list.append(str(Sents_list_raw[i+2]))
                    except:
                        pass
                    try:
                        Sents_list.append(str(Sents_list_raw[i-1]))
                    except:
                        pass
                    try:
                        Sents_list.append(str(Sents_list_raw[i-2]))
                    except:
                        pass
            Sents_list = list(set(Sents_list))
            # Sents_list = sent_splitter(body)
            batch = [{'sentence': str(S)} for S in Sents_list if len(str(S)) < max_sentence]
            res = predictor.predict_batch_json(batch)
            with open(FileName, 'w+') as outfile:
                json.dump(res, outfile)
    with open('check_file_tax_2sents', 'a') as outfile2:
        outfile2.write(str(date) + '\n')

# Still missing:
# Check for files which already exist
# Should be organized by date: which articles are not yet

from joblib import Parallel, delayed
import multiprocessing
n_cores = multiprocessing.cpu_count()
# n_cores = 1
print('There are {0} cores!'.format(n_cores))
Parallel(n_jobs=n_cores)(delayed(process_date)(d)
                            for d in dates_all)
