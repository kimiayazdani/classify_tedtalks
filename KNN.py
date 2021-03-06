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

def validate_data_build(x_tr):
	validate_cut = len(x_tr)//10 + 1
	x_validate = {}
	for i in range(validate_cut):
		m = csc.csc_matrix.toarray(x_tr[i]).flatten()
		x_validate[i]= {}
		for j in range(len(m)):
			if m[j]:
				x_validate[i][j] = m[j]
	print('x_validate created')
	x_train = {}
	for i in range(validate_cut, len(x_tr), ):
		m = csc.csc_matrix.toarray(x_tr[i]).flatten()
		x_train[i-validate_cut]= {}
		for j in range(len(m)):
			if m[j]:
				x_train[i-validate_cut][j] = m[j]
	print('x_test for validation created')
	return x_validate, x_train


def build_train_test_data(df_test = None):
	df_train = read_train_data()
	if df_test is None:
		df_test = read_test_data()
	x_tr = df_train['combined']
	x_te = df_test['combined']
	x_train = {}
	x_test = {}


	for i in range(len(x_tr)):
		m = csc.csc_matrix.toarray(x_tr[i]).flatten()
		x_train[i]= {}
		for j in range(len(m)):
			if m[j]:
				x_train[i][j] = m[j]

	print("x_train created")

	for i in range(len(x_te)):
		m = csc.csc_matrix.toarray(x_te[i]).flatten()
		x_test[i] = {}
		for j in range(len(m)):
			if m[j]:
				x_test[i][j] = m[j]
	print("x_test created")

	y_train = df_train.views.to_list()
	try:
		y_test = df_test.views.to_list()
	except:
		y_test = []
	return x_train, y_train, x_test, y_test

#-----------------------------------------------------------

def dist_calc(tr, ts):
		x = 0

		for key,val in tr.items():
			if key in ts:
				x += (val - ts[key])**2
			else:
				x += val**2
		for key,val in ts.items():
			if key in tr:
				continue
			x += val**2
		return math.sqrt(x)

def predict_one(test, x_train, y_train, k):
	pass




def validate_k():
	x_tr  = read_train_data()
	x_train = x_tr['combined']
	y_train = x_tr.views.to_list()
	k_to_validate = [1, 3, 5, 9]
	validate_cut = len(x_train)//10 + 1
	x_validate, x_train = validate_data_build(x_train)
	y_validate = y_train[:validate_cut]
	y_train = y_train[validate_cut:]
	best_acc, best_k = 0, 0
	for k in k_to_validate:
		y_pred = [predict_one(x_train[i], x_validate, y_validate, k) for i in range(len(x_train))]
		print('validation for k:', k)
		acc = acc_calc(y_pred, y_train)['acc']
		best_acc, best_k = (acc, k) if acc >= best_acc else (best_acc, best_k)
	print("-- best k:", best_k, "best acc:", best_acc, "--")
	return best_k





def train_test(k):
	pass





if __name__ == "__main__":
	best_k = validate_k()
	print("validation is over and now the best k (", best_k, ") will be used on testset.")
	train_test(best_k)