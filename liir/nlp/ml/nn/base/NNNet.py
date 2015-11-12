__author__ = 'quynhdo'
import theano as th
import theano.tensor as T
import timeit
import numpy as np

from liir.nlp.ml.nn.base.Functions import DotActivateFunction
from liir.nlp.ml.nn.base.Functions import SigmoidOutputFunction
from liir.nlp.ml.nn.base.Connection import  Connection
class NNNet:

    '''
        This is a template to make a neural neutron network

    '''


    def __init__(self, layers, cost_function, initial_w=None, initial_b=None, input=None, input_type="matrix", output=None, output_type="matrix", auto_create_connection=False):
        '''

        :param layers: List of layers
        :param cost_function:
        :param initial_w: List of initial weights for connections
        :param initial_b: List of initial bias for connections
        :param input:
        :param input_type: type of input, can be matrix or vector
        :param output:
        :param output_type: type of input, can be matrix or vector
        :param auto_create_connection: True or False, if True, the NNNet init connections between layers automatically
        :return:
        '''
        self.connections=[]
        self.layers=layers
        self.params=[]
        self.cost_function= cost_function

        '''
        input can be a matrix (use for mini batch training) or vector (use for single training)
        '''

        if input is None:
            if input_type == "sparse":
                self.x= th.sparse.csr_matrix('input')

            if input_type == "matrix":
                self.x = T.matrix(name='input')

            if input_type == "vector":
                self.x = T.vector(name='input')
        else:
            self.x = input

        if output is None:
            if output_type == "matrix":
                self.y = T.matrix(name='output')

            if output_type == "vector":
                self.y = T.ivector(name='output')
        else:
            self.y = output

        self.L=None

        if auto_create_connection:
            for i in range(len(layers)-1):
                if initial_w is None:
                    c = self.createConnection(layers[i],layers[i+1])
                    self.connections.append(c)
                    self.params=self.params+c.params

                else:
                    c = self.createConnection(layers[i],layers[i+1], initial_w=initial_w[i], initial_b=initial_b[i])
                    self.connections.append(c)
                    self.params=self.params+c.params


    def setRegularization(self, name="l1", l_lamda = 0.0001):
        self.L = 0
        self.L_lamda = l_lamda
        if name == "l1":

            for conn in self.connections:
                self.L += abs(conn.W).sum()

        if name =="l2":
            for conn in self.connections:
                self.L += abs(conn.W ** 2).sum()



    def createConnection (self, l1, l2,initial_w=None, initial_b=None, af= DotActivateFunction, of = SigmoidOutputFunction, otype=Connection.Output_Type_Real):
        return Connection(scr=l1, dst=l2, activate_func=af, output_func=of, use_bias=l1.useBias , id="c"+l1.id, otype=otype, initial_w=initial_w, initial_b=initial_b)


    def get_connection(self, l1, l2):
        for conn in self.connections:
            if conn.scr == l1 and conn.dst == l2:
                return conn
        return None

    '''
    connect all layers of the Net with the input X
    '''

    def connect(self, x):
        assert len(self.layers) >=2
        self.layers[0].process_input(x)
        for conn in self.connections:
            conn.connect()




    def get_cost_updates(self,learning_rate):

        cost = self.cost_function(self.layers[len(self.layers)-1].output , self.y)

        if self.L is not None:

            cost = cost + self.L_lamda * self.L

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def fit(self, train_data, train_data_label, batch_size, training_epochs, learning_rate):

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
            self.y: train_data_label[index * batch_size: (index + 1) * batch_size]
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





    def predict(self, x):
        t=x
        for i in range(len(self.connections)):
            t = self.connections[i].getOutputValue(t)


        return t

    def getHiddenValue(self, x):
        t=x
        for i in range(len(self.connections)-1):
            t = self.connections[i].getOutputValue(t)

        return t

    def getHidden(self, x):
        t=x
        for i in range(len(self.connections)-1):
            t = self.connections[i].getOutput(t)

        return t