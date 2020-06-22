
# loading packages
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import xlsxwriter
import json 

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
#import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# loading packages

import topic_model_help as tmh
import opiod_database_acess as dsa 



word_list = tmh.expert_words()
date_after = '2017-05-05'
tuple_list =  dsa.keywords_filtering_sql(word_list,date_after)

# reformat keywords column 
tuple_list = [(t[0], t[1], t[2], t[3], "{"+",".join(t[4])+"}") for t in tuple_list]
    
# get ready for the main data frame from SDSC database 
df = tmh.load_df(tuple_list)
df = df[pd.notnull(df['news'])]

print(df.shape)
df = df[pd.isnull(df['language'])]
#TODO add reg language detection 

print(df.shape)
df = df[df["keywords"].apply(tmh.hasKeyWords)]

print(df.shape)

# apply text cleaning section
df['news_clean'] = df['news'].apply(tmh.clean_text)


document = list(df['news_clean'].apply(lambda x: x))
id2word, texts, corpus = tmh.pre_processing(document)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=12, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

for item in lda_model.print_topics():  
    print ( "Topic" , item[0], ":", ", ".join(item[1].split("\"")[1::2]))


import psycopg2
import datetime as datetime

docId_list = df["id"].to_list()
threshhold = 0.1 


with open('./credential.json') as f:
  cre = json.load(f)


connection = psycopg2.connect(user = cre["user"],
                              password = cre["password"],
                              host = "awesome-hw.sdsc.edu",
                              port = "5432",
                              database = "postgres")
cursor = connection.cursor()
for i in range(len(docId_list)):
    print(i)
    docId = docId_list[i]
    t = lda_model[corpus][i]
        
    for pair in t[0]:
        q = '''INSERT INTO topicDoc (docID, topic, prob, modelDate) VALUES ({}, {}, {}, TIMESTAMP '{}')'''    
        if pair[1] > threshhold:
            q =  q.format(docId, pair[0], pair[1], datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")) 
            try:
                cursor.execute(q)
                connection.commit()
            except (Exception, psycopg2.Error) as error :
                print ("Error while connecting to PostgreSQL", error)
