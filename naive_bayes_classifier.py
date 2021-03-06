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


# number of each word appeared in each category / number of whole words in data set
def log_Pxy(grouped_vec, number_whole_words):
    return np.subtract(np.log(grouped_vec), np.log(number_whole_words))


def X_given_Y(df, laplace_smoothing=0.0000001):
    grouped = df.groupby(["views"]).sum()
    grouped['combined'] = grouped['combined'].apply(lambda x: sparce_matrix_to_list(x))
    grouped['description'] = grouped['description'].apply(lambda x: sparce_matrix_to_list(x))
    grouped['title'] = grouped['title'].apply(lambda x: sparce_matrix_to_list(x))

    grouped = grouped.add(laplace_smoothing)  # laplas smoothing 10^-7
    num_of_whole_words = grouped.sum().sum().sum()
    logPxy_com = grouped['combined'].apply(lambda x: log_Pxy(x, num_of_whole_words))
    logPxy_tit = grouped['title'].apply(lambda x: log_Pxy(x, num_of_whole_words))
    logPxy_des = grouped['description'].apply(lambda x: log_Pxy(x, num_of_whole_words))
    logPy = probY(df[['views']])
    logPxy_logPy_com = np.subtract(logPxy_com, logPy)  # P(x,y) / P(y) = log(P(x,y)) - log(P(y))
    logPxy_logPy_tit = np.subtract(logPxy_tit, logPy)
    logPxy_logPy_des = np.subtract(logPxy_des, logPy)
    return (logPxy_logPy_com, logPxy_logPy_des, logPxy_logPy_tit)


# calculate probability of each category
def probY(y):
    size_of_each_category = y.groupby('views').size()
    size_of_whole_data_set = len(y)
    yPrior = np.subtract(np.log(size_of_each_category), np.log(size_of_whole_data_set))
    yPrior = yPrior.to_numpy()
    return yPrior


def max_p(predictions, prefix):
    prediction = 0
    if (predictions[prefix + ' P 1'] > predictions[prefix + ' P -1']):
        prediction = 1
    else:
        prediction = -1
    return prediction


def predict(df_test, logPxy_logPy_com, logPxy_logPy_des, logPxy_logPy_tit):
    df_test['description'] = df_test['description'].apply(lambda x: sparce_matrix_to_list(x))
    df_test['title'] = df_test['title'].apply(lambda x: sparce_matrix_to_list(x))
    df_test['combined'] = df_test['combined'].apply(lambda x: sparce_matrix_to_list(x))

    df_test['des P 1'] = df_test['description'].apply(lambda x: np.multiply(x, logPxy_logPy_des.loc[1]).sum())
    df_test['des P -1'] = df_test['description'].apply(lambda x: np.multiply(x, logPxy_logPy_des.loc[-1]).sum())
    df_test['tit P 1'] = df_test['title'].apply(lambda x: np.multiply(x, logPxy_logPy_tit.loc[1]).sum())
    df_test['tit P -1'] = df_test['title'].apply(lambda x: np.multiply(x, logPxy_logPy_tit.loc[-1]).sum())
    df_test['com P 1'] = df_test['combined'].apply(lambda x: np.multiply(x, logPxy_logPy_com.loc[1]).sum())
    df_test['com P -1'] = df_test['combined'].apply(lambda x: np.multiply(x, logPxy_logPy_com.loc[-1]).sum())

    df_test['tit prediction'] = df_test.apply(lambda row: max_p(row, 'tit'), axis=1)
    df_test['des prediction'] = df_test.apply(lambda row: max_p(row, 'des'), axis=1)
    df_test['com prediction'] = df_test.apply(lambda row: max_p(row, 'com'), axis=1)

    return df_test


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


def bayesClassifier_hyper_par(laplace_smoothing_lower=0.2, laplace_smoothing_upper=2, step=0.1):
    # path to read data from
    pathSrc = str(pathlib.Path().absolute())

    # read train data here
    df_train, vocabulary = read_train_data(pathSrc)
    print('read data.')
    # read test data here
    df_test = read_test_data(pathSrc)
    print('read test data\n')

    laplace_accuracy = []

    start = int(round(tm.time() * 1000))
    laplace = laplace_smoothing_lower
    number_of_parameter_checking = int(np.floor((laplace_smoothing_upper - laplace_smoothing_lower) / step)) + 1
    process = 0
    while laplace < laplace_smoothing_upper:
        process += 1
        print('process {} out of {}'.format(process, number_of_parameter_checking))
        print("ACCURACY BASE ON LAPLACE= {}:\n\t".format(laplace))
        bayesClassifier(pathSrc, laplace)
        laplace += step

        print('------------------------')

    end = int(round(tm.time() * 1000))

    print('Hyper Parameter Laplace END.\n time of Computing is:\n\t{} ms'.format(end - start))
    print('arguments: (laplace_smoothing_lower = {}, laplace_smoothing_upper= {}, step = {})'.format(
        laplace_smoothing_lower, laplace_smoothing_upper, step))
    print('\n(laplace, its accuracy on data):\n########################################')
    for tuples in laplace_accuracy:
        print("\t\t", tuples)


def main():
    path = str(pathlib.Path().absolute())
    #    bayesClassifier(path)
    bayesClassifier_hyper_par()

# main()
