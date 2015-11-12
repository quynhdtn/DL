__author__ = 'quynhdo'
import theano as th
import numpy as np
import theano.tensor as T

class LinearClassifier(object):

    def __init__(self, n_in, n_out):
        """

        :param n_in: dimension of input
        :param n_out: dimension of output
        :return:
        """
        self.trained = False
        self.nIn = n_in
        self.nOut = n_out
        self.W = th.shared(value= np.zeros((n_in,n_out), dtype=th.config.floatX), name="W")  # parameters init
        self.b = th.shared(value = np.zeros(n_out, dtype=th.config.floatX), name="b" ) # bias
        self.params = [self.W, self.b]


    def train(self, x,y ):
        '''
        :param x: feature vector of the instance
        :param y: label
        :return:  Weight vector
        '''

        raise NotImplementedError("The method hasn't been implemented!")

    def transform_function(self, x):
        return x.transpose()

    def score(self, x):
        return T.dot( self.W , self.transform_function(x)) + self.b

    def getLabel(self, x):
        return T.argmax(self.score(x),axis=0)






