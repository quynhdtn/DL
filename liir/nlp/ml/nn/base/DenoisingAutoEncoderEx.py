from liir.nlp.ml.nn.base.DenoisingAutoEncoder import DenoisingAutoEncoder

__author__ = 'quynhdo'
import theano as th
from liir.nlp.ml.nn.base.AutoEncoder import AutoEncoder
from liir.nlp.ml.classifiers.linear.logistic import load_data
try:
    import PIL.Image as Image
except ImportError:
    import Image
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images
from liir.nlp.ml.nn.base.Layer  import Layer,DenoisingLayer
from liir.nlp.ml.nn.base.NNNet  import NNNet
from liir.nlp.ml.nn.base.Functions import CrossEntroyCostFunction
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import numpy as np

import timeit

class DenoisingAutoEncoderEx(DenoisingAutoEncoder):
    def __init__(self,nIn=700, nHidden=500, corruption_level=0.1,id="", initial_w=None, initial_b=None,input=None, input_type="matrix", output=None, output_type="matrix",cost_function=CrossEntroyCostFunction, full_x=None, knowledge = None):

        DenoisingAutoEncoder.__init__(self,nIn, nHidden, corruption_level, id, initial_w, initial_b, input, input_type, output, output_type, cost_function)

        self.full_x = full_x

        if knowledge != None:
            self.knowledge = knowledge
        else:
            self.knowledge = None

        self.L=None

    def get_cost_updates(self,learning_rate):
        cost = self.cost_function(self.layers[len(self.layers)-1].output , self.y)
        if self.L is not None:
            cost = cost + self.L

        if self.knowledge is not None:
            for p in self.knowledge.eval()['pair']:
                h1= self.getHidden(self.full_x[p[0]])
                h2= self.getHidden(self.full_x[p[1]])
                cost = cost + T.mean ((h1-h2)**2)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def fit(self, train_data,  batch_size, training_epochs, learning_rate, ):

        index = T.lscalar()
      #  x = T.matrix('x')
     #   if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_SoftMax):
     #       y = T.ivector('y')

      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Binary):
      #      y = T.iscalar('y')
      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Real):
       # y = T.matrix('y')

        cost,updates=self.get_cost_updates(learning_rate)
        train_da = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            self.x: train_data[index * batch_size: (index + 1) * batch_size],
            self.y: train_data[index * batch_size: (index + 1) * batch_size],
#            self.full_x: train_data
#            self.knowledge : train_data
            }
        )
        n_train_batches = (int) (train_data.get_value(borrow=True).shape[0] / batch_size)

     #   n_train_batches =2
        start_time = timeit.default_timer()
        for epoch in range(training_epochs):
        # go through trainng set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(train_da(batch_index))

            print ('Training epoch %d, cost ' % epoch, np.mean(c))

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)
        print('Training time: %.2fm' % ((training_time) / 60.))


if __name__ == "__main__":
    dataset='/Users/quynhdo/Downloads/mnist.pkl'
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    knowledge = th.shared( {"pair":[(0,1),(3,4)]})
    ae = DenoisingAutoEncoderEx(28*28,500, corruption_level=0.3, knowledge=knowledge)
   # ae.fit(train_set_x, 20, 5,0.1, {'pair':[(0,1), (1,2)]})

    ae.full_x= train_set_x

    ae.fit(train_set_x, 20, 5,0.1)

