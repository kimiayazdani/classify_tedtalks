from naive_bayes_classifier import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from accrecprec import *
from sklearn.metrics import precision_recall_fscore_support


def random_forest(mode="combined"):
    path_src = str(pathlib.Path().absolute())
    df_train, vocabulary = read_train_data(path_src)
    df_train[mode] = df_train[mode].apply(lambda x: sparce_matrix_to_list(x))
    arr_train = []
    for item in df_train[mode]:
        for elem in item:
            arr_train.append(elem)

    array = np.array(arr_train)