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
    x_train = np.reshape(array, (len(df_train[mode]), len(df_train[mode][0])))
    y_train = df_train["views"]
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    df_test = read_test_data(path_src)
    df_test[mode] = df_test[mode].apply(lambda x: sparce_matrix_to_list(x))
    arr_test = []
    for item in df_test[mode]:
        for elem in item:
            arr_test.append(elem)

    arr = np.array(arr_test)
    x_test = np.reshape(arr, (len(df_test[mode]), len(df_test[mode][0])))
    y_test = df_test["views"]
    y_pred = rfc.predict(x_test)
    # acc_calc(y_pred, y_test)
    # print(precision_recall_fscore_support(y_test, y_pred))
    print(y_pred)
    return y_pred
