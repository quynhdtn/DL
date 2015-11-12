

__author__ = 'quynhdo'
from liir.nlp.ml.nn.base.DenoisingAutoEncoder import DenoisingAutoEncoder
from liir.nlp.ml.classifiers.linear.logistic import load_data
import theano as th
try:
    import PIL.Image as Image
except ImportError:
    import Image
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images
from liir.nlp.ml.nn.base.MLP import MLP
from liir.nlp.ml.nn.base.Layer import Layer
import theano.tensor as T
import timeit
import numpy as np
from liir.nlp.ml.nn.base.Connection import Connection
from liir.nlp.ml.nn.base.Functions import SquaredErrorCostFunction, SigmoidOutputFunction, \
    NegativeLogLikelihoodCostFunction

class StackDenoisingAutoEncoderEx:


    def __init__(self, numInput, numHiddens, numOutput, corruption_level, numExtend=None, input=None, extend_input=None, input_type="matrix", extend_input_type="matrix", output=None, output_type="matrix",id=""):

        layers=[]
        ilayer = Layer(numNodes=numInput, ltype = Layer.Layer_Type_Input, id=id+"0")  # input layer
        layers.append(ilayer)

        if numExtend is None:
            for i in  range(len(numHiddens)):
                hlayer = Layer(numNodes=numHiddens[i], ltype = Layer.Layer_Type_Hidden, id=id+str(i))
                layers.append(hlayer)
        else:
            for i in  range(len(numHiddens)-1):
                hlayer = Layer(numNodes=numHiddens[i], ltype = Layer.Layer_Type_Hidden, id=id+str(i))
                layers.append(hlayer)
            hlayer = Layer(numNodes=numHiddens[len(numHiddens)-1], ltype = Layer.Layer_Type_Hidden, id=id+str(i), extendNodes=numExtend)
            layers.append(hlayer)

        # construct the first DA:


        if input is None:
            if input_type == "matrix":
                self.x = T.matrix(name='input')

            if input_type == "vector":
                self.x = T.vector(name='input')

            if input_type == "sparse":
                self.x = th.sparse.csr_matrix(name='input')
        else:
            self.x = input


        if output is None:
            if output_type == "matrix":
                self.y = T.matrix(name='output')

            if output_type == "vector":
                self.y = T.vector(name='output')
        else:
            self.y = output

        if extend_input is None:
            if extend_input_type == "matrix":
                self.x_extend = T.matrix(name='extend_input')

            if extend_input_type == "vector":
                self.x = T.vector(name='extend_input')

            if extend_input_type == "sparse":
                self.x_extend = th.sparse.csr_matrix('extend_input')

        else:
            self.x_extend = extend_input

        self.mlp = MLP(layers, numOutput, activate_function=SigmoidOutputFunction, cost_function=NegativeLogLikelihoodCostFunction, input=self.x, extend_input=self.x_extend, input_type=input_type)


        dA= DenoisingAutoEncoder(layers[0].size, layers[1].size, initial_w=self.mlp.connections[0].W, initial_b=self.mlp.connections[0].b,  input=self.x, id="da0", corruption_level=corruption_level[0])
 #       dA.W = self.mlp.connections[0].W
 #       dA.b= self.mlp.connections[0].b
 #       dA.connect(dA.x)
        self.dAs = [dA]

        for i in range(2, len(layers)-1):
            dA= DenoisingAutoEncoder(layers[i-1].size, layers[i].size,initial_w=self.mlp.connections[i-1].W, initial_b=self.mlp.connections[i-1].b, input=self.mlp.layers[i-1].output, id="da"+str(i-1), corruption_level=corruption_level[i-1])
         #   dA.W = self.mlp.connections[i-1].W
         #   dA.b= self.mlp.connections[i-1].b
            self.dAs.append(dA)

        print(len(self.dAs))

    def preTraining(self, train_data, batch_size, training_epochs, learning_rate):
        for dA in self.dAs:
            dA.connect(dA.x)

        index = T.lscalar()
      #  x = T.matrix('x')
     #   if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_SoftMax):
     #       y = T.ivector('y')

      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Binary):
      #      y = T.iscalar('y')
      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Real):
       # y = T.matrix('y')


        pretrainfns= []
        i=0
        for dA in self.dAs:
            print(i)
            cost,updates=dA.get_cost_updates(learning_rate)
            train_da = th.function(
            [index],
            cost,
            updates=updates,
            givens={
                self.x: train_data[index * batch_size: (index + 1) * batch_size],
                }
            )
            pretrainfns.append(train_da)
            i+=1

        for train_da in pretrainfns:
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



    '''

    def __init__(self, *dA, size_output_layer=100,  activate_function=None, cost_function=SquaredErrorCostFunction):
        self.dAs= dA
        layers =[]
        for i in range(len(self.dAs)):
            if i==0:
                #the first layer:
                layers.append(self.dAs[0].layers[0])
                layers.append(self.dAs[0].layers[1])
            else:
                layers.append(self.dAs[i].layers[1])
        self.mlp = MLP(layers=layers,size_output_layer=size_output_layer,activate_function= activate_function, cost_function= cost_function)

    def preTrain(self, train_data, batch_size, training_epochs, learning_rate):

        ### pretrain step
        td= train_data
        for i in range(len(self.dAs)):
            print ("Pre-train model %d..." % i)
            self.dAs[i].fit(td,batch_size, training_epochs, learning_rate)
            td=th.shared(self.dAs[i].connections[0].getOutputValue(td))

        ###
        for i in range(len(self.dAs)):
            self.mlp.params.remove(self.mlp.connections[i].W)
            self.mlp.params.remove(self.mlp.connections[i].b)
            self.mlp.connections[i].W = self.dAs[i].connections[0].W
            self.mlp.connections[i].b = self.dAs[i].connections[0].b
            self.mlp.params.append(self.mlp.connections[i].W)
            self.mlp.params.append(self.mlp.connections[i].b)

        print("Finish pre-training!")

    def fit( self, train_data, train_data_label, batch_size, training_epochs, learning_rate):
        self.mlp.fit(train_data, train_data_label, batch_size, training_epochs, learning_rate)

    '''

if __name__ == "__main__":
    dataset='/Users/quynhdo/Downloads/mnist.pkl'
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    test_set_x, test_set_y = datasets[1]



    sda= StackDenoisingAutoEncoderEx(28*28-2,numHiddens=[1000, 1000, 1000],numOutput=10, numExtend=2, corruption_level=[0.1,0.2,0.3])
    sda.preTraining(th.shared(train_set_x.eval()[:,0:28*28-2]), 20, 2,0.01)

    sda.mlp.fit(th.shared(train_set_x.eval()[:,0:28*28-2]),th.shared(train_set_x.eval()[:,28*28-2:28*28]), train_set_y, 20,1,0.1)

    y_pred = sda.mlp.predict(test_set_x.eval()[:,0:28*28-2], test_set_x.eval()[:,28*28-2:28*28])
    print(y_pred)
    #print (test_set_y)
    from sklearn import metrics
    print (metrics.f1_score(test_set_y.eval(),y_pred))



    # we need a function to convert Y set from vector to matrix