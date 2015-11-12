__author__ = 'quynhdo'
import theano as th
import theano.tensor as T
import numpy as np

import theano.sparse
import scipy

#### activate functions

def DotActivateFunction(x, W, b):
        if b !=None:
            return th.dot(x, W) + b
        else:
            return th.dot(x,W)

def DotActivateFunctionExtended( x, x_e, W, b):

        if b != None:
        #    return th.dot(T.concatenate([x,x_e], axis=1), W) + b
            if isinstance(x, (scipy.sparse.spmatrix, np.ndarray, tuple, list)):
                if theano.sparse.basic._is_sparse(x_e):
                    xx = th.sparse.hstack((theano.sparse.csr_from_dense(x),x_e))
                    return th.sparse.structured_dot(xx, W) + b
            else:
                if theano.sparse.basic._is_sparse_variable(x_e):
                    xx = th.sparse.hstack((theano.sparse.csr_from_dense(x),x_e))
                    return th.sparse.structured_dot(xx, W) + b

            return th.dot(T.concatenate([x,x_e], axis=1), W) + b

        else:
            return T.dot(x,W)

def NoneActivateFunction(self, x, W, b):
        return x



#### output functions
SigmoidOutputFunction = T.nnet.sigmoid
SoftmaxOutputFunction = T.nnet.softmax
TanhOutputFunction = T.tanh
def NoneOutputFunction (x):
    return x



#### cost functions
def NegativeLogLikelihoodCostFunction(o, y, net=None):
        return -T.mean(T.log(o)[T.arange(y.shape[0]), y])



def SquaredErrorCostFunction(o,y, net=None):
        return T.mean((o-y) ** 2)

def CrossEntroyCostFunction(o, y, net=None):
    L = - T.sum(y * T.log(o) + (1 - y) * T.log(1 - o), axis=1)

    cost = T.mean(L)
    return cost



#### functions to process input for the input layer


def NoneProcessFunction(x, *args): return x
