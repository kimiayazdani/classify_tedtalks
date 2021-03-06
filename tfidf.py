import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc
import pathlib
import pickle


def read_test_data(pathSrc):
    df = pd.read_csv(pathSrc + "/test_preprocessed.csv")
    df = df.fillna('')
    df['combined'] = df['description'] + ' ' + df['title']
    return df


def read_train_data(pathSrc):
    df = pd.read_csv(pathSrc + "/train_preprocessed.csv")
    df = df.fillna('')
    df['combined'] = df['description'] + ' ' + df['title']
    voc = np.load(pathSrc + "/voc.npy", allow_pickle=True).tolist()
    return (df, voc)


def tf_idf(df, voc, idf=None, mode='train'):
    vectorizer_tit = CountVectorizer(token_pattern='\w+', vocabulary=voc)
    vectorizer_des = CountVectorizer(token_pattern='\w+', vocabulary=voc)
    vectorizer_com = CountVectorizer(token_pattern='\w+', vocabulary=voc)

    # fit vectorizer on data(text:srting) -> vector of features
    vec_fit_tit = vectorizer_tit.fit_transform(df['title'])
    vec_fit_des = vectorizer_des.fit_transform(df['description'])
    vec_fit_com = vectorizer_com.fit_transform(df['combined'])

    # count each word
    counts_tit = np.array(vec_fit_tit.sum(axis=0)).flatten().tolist()
    counts_des = np.array(vec_fit_des.sum(axis=0)).flatten().tolist()
    counts_com = np.array(vec_fit_com.sum(axis=0)).flatten().tolist()

    # get each uniq word in data
    words_tit = vectorizer_tit.get_feature_names()
    words_des = vectorizer_des.get_feature_names()
    words_com = vectorizer_com.get_feature_names()

    # dictionary of word and collection frequency
    df_tit = pd.Series(counts_tit, index=words_tit).to_dict()
    df_des = pd.Series(counts_des, index=words_des).to_dict()
    df_com = pd.Series(counts_com, index=words_com).to_dict()

    return (df, idf)


def write_to_file(file, addr):
    with open(addr + '_pre_ntn.pickle', 'wb') as handle:
        pickle.dump(file, handle)


def read_from_file(addr):
    with open(addr + '_pre_ntn.pickle', 'rb') as handle:
        return pickle.load(handle)


def main():
    database_path = str(pathlib.Path().absolute())
    df_train, voc = read_train_data(database_path)
    print("Read Train Data")
    df_test = read_test_data(database_path)
    print("Read Test Data")

    df_train, idf = tf_idf(df_train, voc, mode='train')
    print("Train tfidf Done")
    df_test, _ = tf_idf(df_test, voc, idf, mode='test')
    print("Test tfidf Done")

    write_to_file(df_train, addr=database_path + "/train")
    write_to_file(df_test, addr=database_path + "/test")


main()
