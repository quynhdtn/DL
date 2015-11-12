__author__ = 'quynhdo'

import theano as th
import numpy as np
import theano.tensor as T
from liir.nlp.ml.nn.base.Layer import Layer
from theano.tensor.shared_randomstreams import RandomStreams
from liir.nlp.ml.nn.base.Functions import SigmoidOutputFunction,TanhOutputFunction,SoftmaxOutputFunction, \
    DotActivateFunctionExtended
# implement the connection between two layers
class Connection:


    Output_Type_Real="real"
    Output_Type_Binary="binary"
    Output_Type_SoftMax="softmax"



    def __init__(self, scr, dst, activate_func, output_func, use_bias=True, id="", initial_w = None, initial_b=None,otype="real" ):
        self.scr = scr  # source layer
        self.dst = dst  # destination layer
        if not self.scr.isExtended:
            self.activate_func = activate_func # transfer function
        else:
            self.activate_func= DotActivateFunctionExtended
        self.output_func = output_func  # activate function
        self.otype=otype
   #     self.W=th.shared(value=np.zeros((scr.size, dst.size), dtype=th.config.floatX), name="W" + id, borrow=True)
        self.rng = np.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        s=scr.size
        if self.scr.isExtended:
                s= scr.size+ scr.size_extend
        if initial_w != None:
            self.W = initial_w
        else:
            self.W=None


            if output_func is SigmoidOutputFunction or output_func is SoftmaxOutputFunction:
                '''
                self.W = th.shared(value=np.asarray(
                        self.rng.uniform(
                            low=-4 * np.sqrt(6. / (s + dst.size)),
                            high=4 * np.sqrt(6. / (s + dst.size)),
                            size=(s, dst.size)
                        ),
                        dtype=th.config.floatX
                    ),name='W'+id, borrow=True)
                '''

                self.W = th.shared(value=np.asarray(
                        self.rng.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(s, dst.size)
                        ),
                        dtype=th.config.floatX
                    ),name='W'+id, borrow=True)

            if output_func is TanhOutputFunction :
                self.W= th.shared(value = np.asarray(
                        self.rng.uniform(
                            low=- np.sqrt(6. / (s + dst.size)),
                            high= np.sqrt(6. / (s + dst.size)),
                            size=(s, dst.size)
                        ),
                        dtype=th.config.floatX
                    ),name='W'+id, borrow=True)

    #    if self.W is None:
    #        self.W=th.shared(value=np.zeros((scr.size, dst.size), dtype=th.config.floatX), name="W" + id, borrow=True)


        self.params=[self.W]
        if use_bias:
            if initial_b != None:
                self.b=initial_b
            else:
                self.b = th.shared(value=np.zeros(dst.size), name="b" + id, borrow=True)
            self.params.append(self.b)
        else:
            self.b = None

    # when connect is called, the output of dst layer is calculated
    def connect(self):
        if self.scr.isExtended:
            self.dst.output = self.output_func(self.activate_func(self.scr.output, self.scr.extend,  self.W, self.b))
        else:
            self.dst.output = self.output_func(self.activate_func(self.scr.output, self.W, self.b))
      #  if self.dst.ltype == Layer.Layer_Type_Output:
      #      self.computeOutput(self.dst.output)




    def computeOutput(self,y_pred):

        if self.otype == Connection.Output_Type_Binary:
            self.dst.output = T.round(y_pred)

        if self.otype == Connection.Output_Type_SoftMax:
            self.dst.output = T.argmax(y_pred, axis=1)



    def getOutputValue(self, x):
        y_pred = self.output_func(self.activate_func(x, self.W, self.b)).eval()
        if self.dst.ltype == Layer.Layer_Type_Output:
            if self.otype == Connection.Output_Type_Binary:
                return T.round(y_pred).eval()

            if self.otype == Connection.Output_Type_SoftMax:
                return T.argmax(y_pred, axis=1).eval()

            if self.otype == Connection.Output_Type_Real:
                return y_pred

        else:
            return y_pred

    def getOutput(self, x):
        return self.output_func(self.activate_func(x, self.W, self.b))



    def getOutputValueExtended(self, x, x_extend):
        y_pred = self.output_func(self.activate_func(x, x_extend, self.W, self.b)).eval()
        if self.dst.ltype == Layer.Layer_Type_Output:
            if self.otype == Connection.Output_Type_Binary:
                return T.round(y_pred).eval()

            if self.otype == Connection.Output_Type_SoftMax:
                return T.argmax(y_pred, axis=1).eval()

            if self.otype == Connection.Output_Type_Real:
                return y_pred

        else:
            return y_pred
