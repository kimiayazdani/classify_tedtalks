# Naive Bayes Classifier
# Bag of Words

import numpy as np
import math
import pandas as pd
import time as tm
from scipy.sparse import csc
import pathlib
import pickle


def read_train_data(pathSrc):
    with open(pathSrc + '/train_pre_ntn.pickle', 'rb') as handle:
        file = pickle.load(handle)
    voc = np.load(pathSrc + "/voc.npy", allow_pickle=True).tolist()
    return (file, voc)


def read_test_data(addr):
    with open(addr + '/Data_from_before/ted_talks_pre_ntn.pickle', 'rb') as handle:
        return pickle.load(handle)


def sparce_matrix_to_list(sparce_matrix):
    return csc.csc_matrix.toarray(sparce_matrix).flatten()


def accuracyCalc(y_pred, y_gold):
    num_correct = 0
    for i in range(len(y_gold)):
        if y_pred[i] == y_gold[i]:
            num_correct += 1
    accuracy = num_correct / len(y_gold)
    return accuracy


def bayesClassifier(pathSrc, laplace_smoothing):
    # read train data here
    df, vocabulary = read_train_data(pathSrc)
    print('read data.')

    start = int(round(tm.time() * 1000))

    laplace_smoothing = laplace_smoothing
    x_given_y_com, x_given_y_des, x_given_y_tit = X_given_Y(df, laplace_smoothing=laplace_smoothing)
    print('x given y computation complete.\n\tLAPLACE SMOOTHING = {}'.format(laplace_smoothing))

    py = probY(df[['views']])

    end = int(round(tm.time() * 1000))
    print(
        'y prior probability computation complete.\n\nlearning time (including X_Given_Y and probY calculation:\n\tLEARNING TIME = {} ms'.format(
            end - start))

    # read test data
    df_test = read_test_data(pathSrc)
    print('read test data\n')

    # make predictions
    start = int(round(tm.time() * 1000))
    y_pred = predict(df_test, x_given_y_com, x_given_y_des, x_given_y_tit)
    end = int(round(tm.time() * 1000))
    print('prediction complete\n\tPREDICTING TIME = {} ms'.format(end - start))

    # print accuracy
    print('\n\tACCURACY BASE ON TITLE= : {}'.format(accuracyCalc(y_pred['tit prediction'], y_pred['views'])))
    print('\n\tACCURACY BASE ON DESCRIPTION= : {}'.format(accuracyCalc(y_pred['des prediction'], y_pred['views'])))
    print(
        '\n\tACCURACY BASE ON TITLE+DESCRIPTION= : {}'.format(accuracyCalc(y_pred['com prediction'], y_pred['views'])))