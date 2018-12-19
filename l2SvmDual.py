#!/bin/python

# Train an L2-regularized Dual L2-SVM classifier

# (C) John T. Halloran, 2017

from __future__ import division
import sys
sys.path.append('/home/jthalloran/soft/liblinear-2.11/python')
from liblinearutil import svm_read_problem
from cvxpy import *
import numpy as np
import timeit
from scipy.sparse import csr_matrix

dset = '/home/jthalloran/classificationData/ijcnn1.tr'
# dset = 'rcv1_train.binary'
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
        X[i,j-1] = x[j]
Y = np.array(Y)
del X0[:]

start_time = timeit.default_timer()
K = []
K = Y[:, None] * X

elapsed = timeit.default_timer() - start_time
print "%f seconds to load data" % (elapsed)

start_time = timeit.default_timer()

K = np.dot(K, K.T)
C = 4.0
K[np.diag_indices(n)] += 1 / (2 * C)
v = Variable(n)
loss = sum_entries(v) - 0.5 * quad_form(v, K)
prob = Problem(Maximize(loss), [v >= 0])

elapsed = timeit.default_timer() - start_time
print "%f seconds spent formulating problem" % (elapsed)

# Solve problem and time it
start_time = timeit.default_timer()
# prob.solve()
prob.solve(solver=SCS, verbose=True, eps = 1e-5)
elapsed = timeit.default_timer() - start_time

print "%f seconds to train dual L2-SVM with L2 regularization" % (elapsed)

# Convert dual-parameters to primal weights
z = (Y[:, None] * X).T * v.value

print z.shape

# training error
h = np.asarray(np.sign(X.dot(z))).reshape(-1)
train_error = float(sum(h != np.sign(Y))) / float(n)
print "%f train accuracy" % (1.0 - train_error)

f = open('l2SvmDual_learnedWeights.txt', 'w')
for i in v.value:
    f.write("%f\n" % i)
f.close()
