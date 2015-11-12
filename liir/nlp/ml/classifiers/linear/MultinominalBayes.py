__author__ = 'quynhdo'


# the multinominal bayes algorithm - features are 1/0
# y^= argmax P(y) * P(x/y) = argmax log (P(y)) + log (P(x|y))
# = argmax log (P(y)) + log ( P(x1|y) * P P(x2|y) *... *  P(xJ|y))
# = argmax log (P(y)) + log (P(x1|y)) + ... + log (P(xJ|y))
from liir.nlp.ml.classifiers.linear.LinearClassifier import LinearClassifier
import numpy as np
import theano as th
import theano.tensor as T

class MultinominalBayes(LinearClassifier):

    def __init__(self, n_in, n_out):
        super().__init__( n_in, n_out)

        self.cache_f = th.shared(np.zeros((n_out, n_in)), th.config.floatX)  # caching feature /class information
        self.cache_c = th.shared(np.zeros(n_out), th.config.floatX)   # caching class information


    def train(self, x,y):

    #    assert  len(x) == len(y)
    #    assert len(x) >0
    #    assert  len(x[0]) == self.nIn
        for i in range(self.nOut):
            for j in range (self.nIn):

                    self.cache_f.get_value(borrow=True)[y.get_value()[i]][j] += x.get_value(borrow=True)[i][j]

            self.cache_c.get_value(borrow=True)[y.get_value()[i]]+=1

        ### update w

        print(np.sum(self.cache_c.get_value(borrow=True)))


        self.b.set_value(np.log( self.cache_c.get_value(borrow=True) / np.sum(self.cache_c.get_value(borrow=True))  ))

        count = np.sum(self.cache_f.get_value(borrow=True), axis=1)

        for i in range (self.nOut):

                self.W.get_value(borrow=True)[i]=np.log(self.cache_f.get_value()[i] + 1/count[i] + self.nIn)




    def transform_function(self, x):
        return x


import time
t0 = time.time()
X= th.shared(np.asmatrix([[1,2,3],[0,0,2]], dtype= th.config.floatX))
y = th.shared(np.asarray([0,1],dtype= th.config.floatX))

mc = MultinominalBayes(3,2)
mc.train(X,y)
print(mc.W.get_value())



print (mc.score(np.asarray([1,2,3], dtype= th.config.floatX)))
t1 = time.time()
print("Looping "
      "took %f seconds" % (t1 - t0))