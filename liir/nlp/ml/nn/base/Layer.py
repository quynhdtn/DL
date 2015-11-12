from liir.nlp.ml.nn.base.Model import Model
from liir.nlp.utils.Enum import Enum

__author__ = 'quynhdo'
import numpy as np

import theano as th
from liir.nlp.ml.nn.base.Functions import NoneProcessFunction
from theano.tensor.shared_randomstreams import RandomStreams


# this class define a Layer in neural network
class Layer:

    Layer_Type_Input="input"
    Layer_Type_Hidden="hidden"
    Layer_Type_Output="output"


    def __init__(self, numNodes, ltype, useBias=True, id="", input_process_func=NoneProcessFunction, extendNodes=None):
        self.ltype = ltype
        self.output = np.zeros(numNodes)
        self.useBias = useBias
        self.size = numNodes
        self.id = id
        self.input_process_func = input_process_func
        self.isExtended=False
        if extendNodes !=None:
            self.extend = np.zeros(extendNodes)
            self.size_extend = extendNodes
            self.isExtended=True

        if ltype == Layer.Layer_Type_Output:
            self.label = None


    def process_input(self, x, *kargs):
        self.output = self.input_process_func(x, kargs)

    def process_extend(self, x_extend, *kargs):
        self.output = x_extend

class DenoisingLayer(Layer):
    def __init__(self, numNodes, ltype, useBias=True, id="",  corruption_level=0.1):
        Layer.__init__(self, numNodes, ltype, useBias, id)
        self.corruption_level = corruption_level

        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.input_process_func = self.denoising_process_function

    def process_input(self, x, *kargs):
        self.output = self.input_process_func(x)


    def denoising_process_function(self, x):
        return self.theano_rng.binomial(size=x.shape, n=1, p=1 - self.corruption_level, dtype=th.config.floatX) * x

























