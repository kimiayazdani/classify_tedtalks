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
	if (mode == 'train'):
        # calculate idf vector
        N = df.shape[0]
        idf = {}
        for term in df_com.keys():
            idf[term] = np.log(N) - np.log(
                df_com[term] + 0.000001)  # calculate idf based on combined 'title'+'description'.

        idf = csc.csc_matrix(list(idf.values()))  # convert idf values to sparse matrix

    # calculate tfidf vectors
    tfidf_tit_csc = idf.multiply(vec_fit_tit)
    tfidf_des_csc = idf.multiply(vec_fit_des)
    tfidf_com_csc = idf.multiply(vec_fit_com)
    for index, row in df.iterrows():
        df.at[index, 'title'] = tfidf_tit_csc[index]
        df.at[index, 'description'] = tfidf_des_csc[index]
        df.at[index, 'combined'] = tfidf_com_csc[index]

    return (df, idf)