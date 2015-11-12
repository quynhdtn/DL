__author__ = 'quynhdo'

from liir.nlp.ml.nn.base.Layer  import Layer
from liir.nlp.ml.nn.base.NNNet  import NNNet
from liir.nlp.ml.nn.base.Functions import CrossEntroyCostFunction
from liir.nlp.ml.classifiers.linear.logistic import load_data
try:
    import PIL.Image as Image
except ImportError:
    import Image
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images


class AutoEncoder(NNNet):
    def __init__(self, nIn=700, nHidden=500):

        # declare layers
        ilayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Input, id="0")
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden, id="1")
        olayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Output, id="2")

        # declare nnnet
        NNNet.__init__(self, [ilayer, hlayer, olayer], cost_function=CrossEntroyCostFunction, auto_create_connection=True)

        # change parameter constraint
        conn1 = self.connections[0]
        conn2 = self.connections[1]

        self.params.remove(conn2.W)
        conn2.W = conn1.W.T
        self.connect(self.x)

    def fit(self, train_data, batch_size, training_epochs, learning_rate):
        NNNet.fit(self, train_data, train_data, batch_size, training_epochs, learning_rate)
        image = Image.fromarray(
            tile_raster_images(X=self.connections[0].W.get_value(borrow=True).T,
                               img_shape=(28, 28), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
        image.save('test_ae1.png')

class AutoEncoder1:

    def __init__(self, nIn=700, nHidden=500):
        # declare layers
        ilayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Input, id="0")
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden, id="1")
        olayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Output, id="2")

        # declare nnnet
        self.net = NNNet(ilayer, hlayer, olayer, cost_function=CrossEntroyCostFunction)

        # change parameter constraint
        conn1 = self.net.connections[0]
        conn2 = self.net.connections[1]

        self.net.params.remove(conn2.W)
        conn2.W = conn1.W.T


    def fit(self, train_data, batch_size, training_epochs, learning_rate):
        self.net.fit(train_data, train_data, batch_size, training_epochs, learning_rate)
        image = Image.fromarray(
            tile_raster_images(X=self.net.connections[0].W.get_value(borrow=True).T,
                               img_shape=(28, 28), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
        image.save('test_ae.png')

if __name__ == "__main__":
    dataset='/Users/quynhdo/Downloads/mnist.pkl'
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]


    ae = AutoEncoder(28*28,500)
    ae.fit(train_set_x, 20, 5,0.1)
