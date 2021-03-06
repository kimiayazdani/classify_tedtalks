import numpy as np
import math
import pandas as pd
import time as tm
from scipy.sparse import csc
import pathlib
import pickle
from accrecprec import acc_calc


def read_train_data():
    with open('./train_pre_ntn.pickle', 'rb') as handle:
        file = pickle.load(handle)
    return file

def read_test_data():
    with open('./test_pre_ntn.pickle', 'rb') as handle:
        return pickle.load(handle)
