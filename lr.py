#!/bin/python

# Train an L2-regularized Logistic Regression classifier

# (C) John T. Halloran, 2017

from __future__ import division
import sys
sys.path.append('/home/jthalloran/soft/liblinear-2.11/python')
from liblinearutil import svm_read_problem
from cvxpy import *
import numpy as np
import timeit

largeTestSet = True
dset = '/home/jthalloran/classificationData/ijcnn1.tr'
# testset = '/home/jthalloran/classificationData/ijcnn1.t'
# dset = '/home/jthalloran/classificationData/rcv1_train.binary'
# dset = '/home/jthalloran/classificationData/news20.binary'
start_time = timeit.default_timer()
Y, X0 = svm_read_problem(dset)

n =  len(X0)
d = max([max(x) for x in X0 if x])
m = n

print "%d instances, %d features" % (n, d)

X = np.zeros(shape=(n,d))
for i,x in enumerate(X0):
    for j in x:
        X[i][j-1] = x[j]
del X0[:]

elapsed = timeit.default_timer() - start_time
print "%f seconds to load data" % (elapsed)

start_time = timeit.default_timer()
# Form L2-regularized logistic regression
w = Variable(d)
L = [log_sum_exp(vstack(0,-Y[i] * X[i,:] * w)) for i in range(m)]
loss = sum(L)
reg = 0.5 * sum_squares(w)
C = 4.0
prob = Problem(Minimize(C * loss + reg))
elapsed = timeit.default_timer() - start_time
print "%f seconds spent formulating problem" % (elapsed)

# Solve problem and time it
start_time = timeit.default_timer()
# prob.solve()
prob.solve(solver=SCS, verbose=False, eps = 1e-2)
elapsed = timeit.default_timer() - start_time

print "%f seconds to train Logistic Regression with L2 regularization" % (elapsed)

# training error
h = np.asarray(np.sign(X.dot(w.value))).reshape(-1)
train_error = float(sum(h != np.sign(Y))) / float(n)
print "%f train error" % (1.0 - train_error)

# # evaluate testset
# start_time = timeit.default_timer()
# Y, X0 = svm_read_problem(testset)
# n =  len(X0)
# d = max([max(x) for x in X0 if x])
# print "%d instances, %d features" % (n, d)

# if not largeTestSet:
#     X = np.zeros(shape=(n,d))
#     for i,x in enumerate(X0):
#         for j in x:
#             X[i][j-1] = x[j]
#     Y = np.array(Y)
#     del X0[:]

#     h = np.asarray(np.sign(X.dot(w.value))).reshape(-1)
#     test_error = float(sum(h != np.sign(Y))) / float(n)
#     print "%f test error" % (1.0 - test_error)

# else:
#     ########### for larger dataset
#     z = w.value
#     Yhat = np.zeros(n)
#     for i,x in enumerate(X0):
#         Yhat[i] = sum([x[j]*z[j-1] for j in x])
#     Y = np.array(Y)
#     del X0[:]

#     test_error = float(sum(np.sign(Yhat) != np.sign(Y))) / float(n)
#     print "%f test error" % (1.0 - test_error)

# elapsed = timeit.default_timer() - start_time

# print "%f seconds to test classifier" % (elapsed)
