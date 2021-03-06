import nltk
import pandas as pd
import numpy as np
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pathlib

######################functions################################
def case_folding(df):
    df['title'] = df["title"].str.lower()
    df['description'] = df["description"].str.lower()
    return df
	
def contraction(df):
    df['title'] = df['title'].apply(lambda x: contractions.fix(x))
    df['description'] = df['description'].apply(lambda x: contractions.fix(x))
    return df
	
def tokenize(df):
    df['title'] = df['title'].apply(word_tokenize)
    df['description'] = df['description'].apply(word_tokenize)
    return df
	
def filter_stop(text, stop_list):
    return [word for word in text if word not in stop_list]

def delete_stops(df, stop_list):
    df['title'] = df['title'].apply(lambda x: filter_stop(x, stop_list))
    df['description'] = df['description'].apply(lambda x: filter_stop(x, stop_list))
    return df

def lemmatizer(text, wordnet_lemmatizer):
    par_pos_tag = nltk.pos_tag(text)
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    x = []
    for i in range(len(par_pos_tag)):
        lemmatized = wordnet_lemmatizer.lemmatize(par_pos_tag[i][0],pos = tag_dict.get(par_pos_tag[i][1][0].upper() , wordnet.NOUN))
        x.append(lemmatized)
    return x

def lemmatize(df, wordnet_lemmatizer):
    df['title'] = df['title'].apply(lambda x: lemmatizer(x, wordnet_lemmatizer))
    df['description'] = df['description'].apply(lambda x: lemmatizer(x, wordnet_lemmatizer))
    return df
	
def keep_alnum(text):
    return [word for word in text if word.isalnum()]

def remove_punc(df):
    df['title'] = df['title'].apply(lambda x: keep_alnum(x))
    df['description'] = df['description'].apply(lambda x: keep_alnum(x))
    return df

def stemm(df, stemmer):
    df['title'] = df['title'].apply(lambda x: list(map(lambda i: stemmer.stem(i),x)))
    df['description'] = df['description'].apply(lambda x: list(map(lambda i: stemmer.stem(i),x)))
    return df

def count_vectorizer(df):
    vectorizer_tit = CountVectorizer(token_pattern='\w+')
    vectorizer_des = CountVectorizer(token_pattern='\w+')

    #fit vectorizer on data(text:srting)
    vec_fit_tit = vectorizer_tit.fit_transform(df['title'].apply(lambda x: ' '.join(i for i in x)))
    vec_fit_des = vectorizer_des.fit_transform(df['description'].apply(lambda x: ' '.join(i for i in x)))

    #count each word
    counts_tit = np.array(vec_fit_tit.sum(axis=0)).flatten().tolist()
    counts_des = np.array(vec_fit_des.sum(axis=0)).flatten().tolist()

    #get each uniq word in data
    words_tit = vectorizer_tit.get_feature_names()
    words_des = vectorizer_des.get_feature_names()

    #dictionary for creating dataframe
    d_tit = {'word': words_tit, 'Title Frequency': counts_tit}
    d_des = {'word': words_des, 'Description Frequency': counts_des}

    #create dataframes for each of title and description term frequencies. Two seperate df -> number of uniq words
    #in each of the Title and Description is diffrent.
    #create two dataframe then merge two of them together

    df_freq_tit = pd.DataFrame(data=d_tit)
    df_freq_des = pd.DataFrame(data=d_des)
    df_freq_tit = df_freq_tit.set_index('word')
    df_freq_des = df_freq_des.set_index('word')
    #merging two df
    term_freq = pd.concat([df_freq_tit, df_freq_des], axis=1)
    term_freq = term_freq.fillna(0)

    term_freq['Total Frequency'] = term_freq['Description Frequency'] + term_freq['Title Frequency']

    return term_freq.sort_values(by=['Total Frequency'], ascending=False)