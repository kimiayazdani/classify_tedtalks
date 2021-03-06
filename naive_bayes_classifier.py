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
