__author__ = 'quynhdo'

import theano as T
class NNNetComplex:
    def __init__(self, *components,  cost_function, **connection_layer_ids):
        self.components=components


    def fit(self, train_data, train_data_label, batch_size, training_epochs, learning_rate):

        td = train_data
        # train each component
        for i in range(len(self.components)):
            self.components[i].fit(td, )




