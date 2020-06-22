
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
import opiod_database_acess as dsa 
import topic_model_help as tmh


#import stanfordnlp
#nlp = stanfordnlp.Pipeline(processors = "tokenize,mwt,lemma,pos")


word_list = tmh.expert_words()
date_after = '2017-05-05'
tuple_list =  dsa.keywords_filtering_sql(word_list,date_after)

# reformat keywords column 
tuple_list = [(t[0], t[1], t[2], t[3], t[4], "{"+",".join(t[5])+"}") for t in tuple_list]
print(tuple_list[0])    
    
# get ready for the main data frame from SDSC database 
df = tmh.load_df(tuple_list)
df = df[pd.notnull(df['news'])]

df = df[pd.isnull(df['language'])]

print(df.shape)
df = df[df["keywords"].apply(tmh.hasKeyWords)]

df = df[df["title"].apply(tmh.is_news_English)]

print(df.shape)

# apply text cleaning section
df['news_clean'] = df['news'].apply(tmh.clean_text)
df['location'] = None
df = df.apply(tmh.combined_location,axis=1)
