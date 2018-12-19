#!/bin/python

# Train an L2-regularized Primal L2-SVM classifier

# (C) John T. Halloran, 2017

from __future__ import division
import sys
sys.path.append('/home/jthalloran/soft/liblinear-2.11/python')
from liblinearutil import svm_read_problem
from cvxpy import *
import numpy as np
import timeit

dset = '/home/jthalloran/classificationData/ijcnn1.tr'
# dset = '/home/jthalloran/classificationData/rcv1_train.binary'
# dset = '/home/jthalloran/classificationData/news20.binary'

Y, X0 = svm_read_problem(dset)

n =  len(X0)
d = max([max(x) for x in X0 if x])
m = n

print "%d instances, %d features" % (n, d)
start_time = timeit.default_timer()
X = np.zeros(shape=(n,d))
for i,x in enumerate(X0):
    for j in x:
        X[i][j-1] = x[j]
Y = np.array(Y)

del X0[:]

# Form L1-SVM with L2 regularization primal problem
w = Variable(d)
loss = sum_entries(square(pos(1 - mul_elemwise(Y, X*w))))
reg = 0.5 * sum_squares(w)
C = 4.0
prob = Problem(Minimize(C * loss + reg))
elapsed = timeit.default_timer() - start_time
print "%f seconds to formulate L2-SVM with L2 regularization" % (elapsed)

# Solve problem and time it
start_time = timeit.default_timer()
# prob.solve()
prob.solve(solver=SCS, verbose=True, eps = 1e-2)
# prob.solve(solver=CVXOPT, verbose=True, abstol = 1e-2)
# prob.solve(verbose=True)
elapsed = timeit.default_timer() - start_time
print "%f seconds to train L1-SVM with L2 regularization" % (elapsed)

# training error
h = np.asarray(np.sign(X.dot(w.value))).reshape(-1)
train_error = float(sum(h != np.sign(Y))) / float(n)
print "%f train accuracy" % (1.0 - train_error)
