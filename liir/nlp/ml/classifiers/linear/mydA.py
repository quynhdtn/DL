__author__ = 'quynhdo'

import os
import sys
import timeit

import numpy as np

import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from liir.nlp.ml.classifiers.linear.logistic import load_data
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

#### implement a single denoising autoencoder


class gmDA:

    def __init__(self, nIn=700, nHidden=500, corruption_level=0.1, learning_rate=0.013, training_epochs=15, batch_size=20):
        self.nIn=nIn
        self.nHidden=nHidden
        self.corruption_level=corruption_level
        self.learning_rate= learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.W=th.shared(value=np.asarray(
                self.rng.uniform(
                    low=-4 * np.sqrt(6. / (nIn + nHidden)),
                    high=4 * np.sqrt(6. / (nIn + nHidden)),
                    size=(nIn , nHidden)
                ),
                dtype=th.config.floatX
            ),name='W', borrow=True)

        self.b = th.shared(np.zeros(nHidden,dtype=th.config.floatX), name="b", borrow=True) #bias used when working on Hidden layer y
        self.bPrime = th.shared(np.zeros(nIn, dtype=th.config.floatX), name="bPrime", borrow=True)  #bias used when working on Output layer z
        self.WPrime = self.W.T
        self.params=[self.W,self.b,self.bPrime]



    # randomly zero out
    def get_corrupted_input(self, x, corruption_level):
        return self.theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level, dtype=th.config.floatX) * x


    def get_hidden_output(self, x):
        return T.nnet.sigmoid (T.dot(x, self.W) + self.b)


    def get_output_output(self,y):
        return T.nnet.sigmoid(T.dot(y, self.WPrime)+ self.bPrime)

    def get_cost_updates(self, x):

        tilde_x = self.get_corrupted_input(x, self.corruption_level)

        y = self.get_hidden_output(tilde_x)
        z = self.get_output_output(y)

        L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)

        cost = T.mean(L)


        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def train(self, train_data):
        index = T.lscalar()
        x = T.matrix('x')
        cost,updates=self.get_cost_updates(x)
        train_da = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_data[index * self.batch_size: (index + 1) * self.batch_size]
        }
        )

        n_train_batches = (int) (train_data.get_value(borrow=True).shape[0] / self.batch_size)
        start_time = timeit.default_timer()
        for epoch in range(self.training_epochs):
        # go through trainng set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(train_da(batch_index))

            print ('Training epoch %d, cost ' % epoch, np.mean(c))

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print (('The no corruption code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((training_time) / 60.)))
        image = Image.fromarray(
            tile_raster_images(X=self.W.get_value(borrow=True).T,
                               img_shape=(28, 28), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
        image.save('filters_corruption_0.png')



dataset='/Users/quynhdo/Downloads/mnist.pkl'
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]


da = gmDA(28*28,500,0.3,0.1,5,20)
da.train(train_set_x)
