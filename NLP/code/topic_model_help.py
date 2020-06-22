
# loading packages
import json
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

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

STOPWORDS = stop_words

import psycopg2

with open('./credential.json') as f:
  cre = json.load(f)


connection = psycopg2.connect(user = cre["user"],
                              password = cre["password"],
                              host = "awesome-hw.sdsc.edu",
                              port = "5432",
                              database = "postgres")
cursor = connection.cursor()


def clean_text(text):
    """ gets text and apply basic cleanning
        
        text: a string
        
        return: modified initial string
    """
    # text = BeautifulSoup(text, "lxml").text # HTML decoding
    # text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


def expert_words(expert_str= """Drug health harm Overdose death fatal overdose
HIV HCV TB endocarditis infectious disease
Mental 
Emergency
treatment opiate therapy
Health policies disorder  prescription    naloxone Samaritan
enforcement
Drug seizures police crackdowns busting
violent crime arrests incarcerations
cartels
FDA
disorders dependence
prescriptions prescription
"""):
    """ 
        gets expert keywords list 
    
    """
    preferred_list = ["opioid","opioids","overdose","overdoses","drug", "drugs","fentanyl"]
    expert_list =  preferred_list + [x.lower() for x in expert_str.replace("\n", " ").split(" ")]
    expert_list = list(set(expert_list))
    expert_list.remove('')
    return expert_list



def load_df(list_of_tuples, columns = ["id", "publishdate", "news", "language", "title","keywords"]):
    """
    load pandas data.frame from list of tuples (sql results)
    """
    return pd.DataFrame.from_records(list_of_tuples, columns=columns )     

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



# Remove VERB 
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def hasKeyWords(x):
    expert_list = expert_words() 
    s = x[2:-1]
    count = 0 
    s_list = s.split(",")
    for word in s_list:
        if word in expert_list:
            count = count + 1
    if count > 1:
        return True
    else:
        return False

def pre_processing(data):
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en


    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return (id2word, texts, corpus)


def location_extraction(row):
    # list of words to be removed from initial NER result
    list_invalid_location =    ['an','americas','asia','asian','valley', 'beach', 'island', 'city', 'road', 'county',
                                'united states','us', 'u.s','states', 'america','europe', 'eastern europe','usa','ave.',
                                'drive','st.', 'hills', 'uk', 'airport','house','hospital','latin america','central america', 
                                'state','states.','africa',
                                "afghanistan",
                                "albania",
                                "algeria",
                                "andorra",
                                "angola",
                                "antigua and barbuda",
                                "argentina",
                                "armenia",
                                "australia",
                                "austria",
                                "azerbaijan",
                                "bahamas",
                                "bahrain",
                                "bangladesh",
                                "barbados",
                                "belarus",
                                "belgium",
                                "belize",
                                "benin",
                                "bhutan",
                                "bolivia",
                                "bosnia and herzegovina",
                                "botswana",
                                "brazil",
                                "brunei",
                                "bulgaria",
                                "burkina faso",
                                "burundi",
                                "cabo verde",
                                "cambodia",
                                "cameroon",
                                "canada",
                                "central african republic",
                                "chad",
                                "chile",
                                "china",
                                "colombia",
                                "comoros",
                                "congo",
                                "costa rica",
                                "côte d'ivoire",
                                "croatia",
                                "cuba",
                                "cyprus",
                                "czech republic",
                                "democratic republic of the congo",
                                "denmark",
                                "djibouti",
                                "dominica",
                                "dominican republic",
                                "ecuador",
                                "egypt",
                                "el salvador",
                                "equatorial guinea",
                                "eritrea",
                                "estonia",
                                "ethiopia",
                                "fiji",
                                "finland",
                                "france",
                                "gabon",
                                "gambia",
                                "georgia",
                                "germany",
                                "ghana",
                                "greece",
                                "grenada",
                                "guatemala",
                                "guinea",
                                "guinea-bissau",
                                "guyana",
                                "haiti",
                                "honduras",
                                "hungary",
                                "iceland",
                                "india",
                                "indonesia",
                                "iran",
                                "iraq",
                                "ireland",
                                "israel",
                                "italy",
                                "jamaica",
                                "japan",
                                "jordan",
                                "kazakhstan",
                                "kenya",
                                "kiribati",
                                "kuwait",
                                "kyrgyzstan",
                                "laos",
                                "latvia",
                                "lebanon",
                                "lesotho",
                                "liberia",
                                "libya",
                                "liechtenstein",
                                "lithuania",
                                "luxembourg",
                                "madagascar",
                                "malawi",
                                "malaysia",
                                "maldives",
                                "mali",
                                "malta",
                                "marshall islands",
                                "mauritania",
                                "mauritius",
                                "mexico",
                                "micronesia",
                                "monaco",
                                "mongolia",
                                "montenegro",
                                "morocco",
                                "mozambique",
                                "myanmar",
                                "namibia",
                                "nauru",
                                "nepal",
                                "netherlands",
                                "new zealand",
                                "nicaragua",
                                "niger",
                                "nigeria",
                                "north korea",
                                "norway",
                                "oman",
                                "pakistan",
                                "palau",
                                "palestine",
                                "panama",
                                "papua new guinea",
                                "paraguay",
                                "peru",
                                "philippines",
                                "poland",
                                "portugal",
                                "qatar",
                                "republic of moldova",
                                "romania",
                                "russia",
                                "rwanda",
                                "saint kitts and nevis",
                                "saint lucia",
                                "saint vincent and the grenadines",
                                "samoa",
                                "san marino",
                                "são tomé and príncipe",
                                "saudi arabia",
                                "senegal",
                                "serbia",
                                "seychelles",
                                "sierra leone",
                                "singapore",
                                "slovakia",
                                "slovenia",
                                "solomon islands",
                                "somalia",
                                "south africa",
                                "south korea",
                                "south sudan",
                                "spain",
                                "sri lanka",
                                "sudan",
                                "suriname",
                                "swaziland",
                                "sweden",
                                "switzerland",
                                "syrian arab republic",
                                "tajikistan",
                                "tanzania",
                                "thailand",
                                "the former yugoslav republic of macedonia",
                                "timor-leste",
                                "togo",
                                "tonga",
                                "trinidad and tobago",
                                "tunisia",
                                "turkey",
                                "turkmenistan",
                                "tuvalu",
                                "u.k.",
                                "u.s.",
                                "uganda",
                                "ukraine",
                                "united arab emirates",
                                "uruguay",
                                "uzbekistan",
                                "vanuatu",
                                "vatican",
                                "venezuela",
                                "viet nam",
                                "yemen",
                                "zambia",
                                "zimbabwe"
                               ]
    # state dictionary
    dict_state =               {
                                "alabama":"Alabama",
                                "alaska":"Alaska",
                                "arizona":"Arizona",
                                "arkansas":"Arkansas",
                                "california":"California",
                                "colorado":"Colorado",
                                "connecticut":"Connecticut",
                                "delaware":"Delaware",
                                "district of columbia":"District of Columbia",
                                "dc":"District of Columbia",
                                "d.c.":"District of Columbia",
                                "washington":"District of Columbia",
                                "florida":"Florida",
                                "georgia":"Georgia",
                                "hawaii":"Hawaii",
                                "idaho":"Idaho",
                                "illinois":"Illinois",
                                "indiana":"Indiana",
                                "iowa":"Iowa",
                                "kansas":"Kansas",
                                "kentucky":"Kentucky",
                                "louisiana":"Louisiana",
                                "maine":"Maine",
                                "maryland":"Maryland",
                                "massachusetts":"Massachusetts",
                                "michigan":"Michigan",
                                "minnesota":"Minnesota",
                                "mississippi":"Mississippi",
                                "missouri":"Missouri",
                                "montana":"Montana",
                                "nebraska":"Nebraska",
                                "nevada":"Nevada",
                                "new hampshire":"New Hampshire",
                                "new jersey":"New Jersey",
                                "new mexico":"New Mexico",
                                "new york state":"New York",
                                "north carolina":"North Carolina",
                                "north dakota":"North Dakota",
                                "ohio":"Ohio",
                                "oklahoma":"Oklahoma",
                                "oregon":"Oregon",
                                "pennsylvania":"Pennsylvania",
                                "rhode island":"Rhode Island",
                                "south carolina":"South Carolina",
                                "south dakota":"South Dakota",
                                "tennessee":"Tennessee",
                                "texas":"Texas",
                                "utah":"Utah",
                                "vermont":"Vermont",
                                "virginia":"Virginia",
                                "washington state":"Washington",
                                "west virginia":"West Virginia",
                                "wisconsin":"Wisconsin",
                                "wyoming":"Wyoming",
                                "american samoa":"American Samoa",
                                "guam":"Guam",
                                "northern mariana islands":"Northern Mariana Islands",
                                "puerto rico":"Puerto Rico",
                                "u.s. virgin islands":"U.S. Virgin Islands",
                               }
    list_location_find = []
    list_location_count = []
    list_one_location = []
    # split list of location from NER result
    list_location = row['location'][1:-1].split(', ')
    # remove "invalid" locations
    list_location = [i for i in list_location if i.lower() not in list_invalid_location and i != '']
    news_lower = row['news'].lower()
    # dic of first appearance in the news article for each location
    dic_location_find = {i:news_lower.find(i.lower()) for i in list_location}
    # dic of count in the news article for each location
    dic_location_count = {i:news_lower.count(i.lower()) for i in list_location}
    # sort first appearence from low to high
    dic_location_find = {k: v for k, v in sorted(dic_location_find.items(), key=lambda item: item[1])}
    # sort count from high to low
    dic_location_count = {k: v for k, v in sorted(dic_location_count.items(), key=lambda item: item[1], reverse=True)}
    temp_list_location_find_keys = list(dic_location_find)
    temp_list_location_find_values = list(dic_location_find.values())
    temp_list_location_count_keys = list(dic_location_count)
    temp_list_location_count_values = list(dic_location_count.values())
    
    if(len(temp_list_location_find_values) != 0):
        # list of location names with most appearances
        list_highest_freq = []
        for i in range(len(temp_list_location_find_values)):
            if(temp_list_location_count_values[i]==temp_list_location_count_values[0]):
                list_highest_freq.append(temp_list_location_count_keys[i])
        # if a location name is the first word of the article OR highest frequency location name also closer to beginning of the article than other location names
        if (temp_list_location_find_values[0] == 0 or temp_list_location_find_keys[0] in list_highest_freq):
            list_one_location.append(temp_list_location_find_keys[0])
            if(temp_list_location_find_keys[0].lower() in dict_state):
                location_state = dict_state[temp_list_location_find_keys[0].lower()]
                location_city = 'none'
                location_county = 'none'
            elif("County" in temp_list_location_find_keys[0] or "Parish" in temp_list_location_find_keys[0]):
                try:
                    cursor.execute("select CountyOrCountyEquivalent, StateOrTerritory from USCounties where CountyOrCountyEquivalent='%s'" % temp_list_location_find_keys[0])
                    connection.commit()
                except (Exception, psycopg2.Error) as error :
                    print ("Error while connecting to PostgreSQL", error)
                myresult = cursor.fetchall()
                if(len(myresult) == 1):
                    #print(myresult[0][0])
                    location_state = myresult[0][1]
                    location_city = 'none'
                    location_county = myresult[0][0]
                else:   
                    location_state = 'none'
                    location_city = 'none'
                    location_county = 'none'
            else:
                location_state = 'none'
                location_city = 'none'
                location_county = 'none'
        else:
            location_state = 'none'
            location_city = 'none'
            location_county = 'none'
    else:
        location_state = 'none'
        location_city = 'none'
        location_county = 'none'

    #Insert to OpioidDocLoc if location is valid
    if (location_state != 'none' or location_county != 'none' or location_city != 'none'):
        sql = "SELECT * FROM OpioidDocLoc WHERE docId = %(val)s"
        try:
            cursor.execute(sql, { 'val': str(row['id']) })
            connection.commit()
        except (Exception, psycopg2.Error) as error :
            print ("Error while connecting to PostgreSQL", error)
        myresult = cursor.fetchall()
        #Update if the doc id has already existed
        if(len(myresult) == 1):
            sql = "UPDATE OpioidDocLoc SET City = %s, County = %s, State = %s WHERE docId = %s"
            val = (location_city, location_county, location_state, str(row['id']))
        #Insert if the doc id does not exist
        if(len(myresult) == 0):
            sql = "INSERT INTO OpioidDocLoc(docId, City, County, State) VALUES (%s, %s, %s, %s)"
            val = (str(row['id']) , location_city, location_county, location_state)
        try:
            cursor.execute(sql, val)
            connection.commit()
        except (Exception, psycopg2.Error) as error :
            print ("Error while connecting to PostgreSQL", error)

# A helper function to concate neighber location terms together
def combined_location(row):
    st = StanfordNERTagger('../stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '../stanford-ner/stanford-ner.jar',
					   encoding='utf-8')

    text = row["news"]
    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    location = list(filter(lambda x : x[1][1] == 'LOCATION', enumerate(classified_text)))
    location = map(lambda x : (x[0],x[1][0]) , location)
    l = list(location)

    output = list()
    l.sort(key = lambda x: x[0])
    temp = ""
    prevIndex = ""
    
    for item in l:
        if temp == "":
            temp = item[1] 
            prevIndex = item[0]
            continue
        if prevIndex == item[0] - 1:
            temp = temp + " " + item[1]
            prevIndex = item[0]
        else:
            output.append(temp)
            temp = ""
            
    row['location'] = "{"+", ".join(list(set(output)))+"}"
    try:
        location_extraction(row)
    except:
        print("An exception occurred")
    
    return row

def is_news_English(x):
    char_match = re.search("[^\x00-\x7F‘’“”′—–·…€£$]", x)
    #if non English character is in title, label the news as non English news
    if (char_match == None):
        return True
    else:
        print(x)
        return False
    return row
