from liir.nlp.ml.nn.base.Functions import SquaredErrorCostFunction,NegativeLogLikelihoodCostFunction
from liir.nlp.ml.nn.base.Layer import Layer
from liir.nlp.ml.nn.base.MLP import MLP
import math
__author__ = 'quynhdo'


class Factory:

    @staticmethod
    def buildMLPStandard(numInput, numOutput,id=""):
        ilayer = Layer(numNodes=numInput, ltype = Layer.Layer_Type_Input, id=id+"0")  # input layer

        # calculate the number of units for the first hidden layer https://www.researchgate.net/post/In_neural_networks_model_which_number_of_hidden_units_to_select
        nHidden = int( math.sqrt(numInput * (numOutput+2)) + 2 *math.sqrt(numInput / (numOutput+2)) )
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden, id=id+"1")    #hidden layer

        return MLP([ilayer, hlayer], size_output_layer=numOutput, cost_function=NegativeLogLikelihoodCostFunction, id=id)

    @staticmethod
    def buildMLPExtendedHiddenLayer(numInput, numOutput, numExtend=0,id="", input_type="matrix", extend_input_type="sparse" ):
        ilayer = Layer(numNodes=numInput, ltype = Layer.Layer_Type_Input, id=id+"0")  # input layer

        # calculate the number of units for the first hidden layer https://www.researchgate.net/post/In_neural_networks_model_which_number_of_hidden_units_to_select
        nHidden = int( math.sqrt(numInput * (numOutput+2)) + 2 *math.sqrt(numInput / (numOutput+2)) )
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden, id=id+"1", extendNodes=numExtend)    #hidden layer

        return MLP([ilayer, hlayer], size_output_layer=numOutput, cost_function=NegativeLogLikelihoodCostFunction, id=id,
                   input_type=input_type,
                   extend_input_type=extend_input_type)

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    import numpy
    iris = load_iris()
    X= iris.data
    Y = iris.target

    from sklearn import metrics
    import theano as th
    import theano.tensor as T


#    mp=Factory.buildMLPStandard(len(X[0]), len(set(Y)))

    mp=Factory.buildMLPExtendedHiddenLayer(len(X[0])-2, len(set(Y)),numExtend=2, extend_input_type="sparse")
    from sklearn import cross_validation
    import scipy.sparse
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( iris.data, Y, test_size=0.4, random_state=0)
   # X_train = scipy.sparse.csr_matrix( X_train)
#    X_test = scipy.sparse.csr_matrix(y_test)

    mp.fit(th.shared(X_train[:,0 :len(X[0])-2]), th.shared(scipy.sparse.csr_matrix( X_train[:,len(X[0])-2:len(X[0])])), T.cast(th.shared(y_train), 'int32'), 15, 3000, 0.1)


    y_pred = mp.predict(X_test[:,0 :len(X[0])-2], scipy.sparse.csr_matrix(X_test[:,len(X[0])-2:len(X[0])]))



    print (y_pred)

    print (metrics.f1_score(y_test,y_pred))


    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print (metrics.f1_score(y_test,y_pred))


    '''
    from sklearn.datasets import load_iris
    import numpy
    iris = load_iris()
    X= iris.data
    Y = iris.target
    labels = list(set(Y))
    y_standard = numpy.zeros((len(X), len(labels)), dtype="int32")

    for i in range(len(X)):
        y_standard[i][Y[i]]=1



    print (Y)
    from sklearn import metrics
    import theano as th
    import theano.tensor as T


    mp=Factory.buildMLPStandard(len(X[0]), len(set(Y)))
    from sklearn import cross_validation

    X_train, X_test, y_train, y_test = cross_validation.train_test_split( iris.data, y_standard, test_size=0.4, random_state=0)

    mp.fit(th.shared(X_train), T.cast(th.shared(y_train), th.config.floatX), 15, 3000, 0.005)

    y_pred = mp.predict(X_test)


    print (y_pred)

    print (metrics.f1_score(y_test.argmax(axis=1),y_pred))

    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X_train, y_train.argmax(axis=1))
    y_pred=clf.predict(X_test)
    print (metrics.f1_score(y_test.argmax(axis=1),y_pred))
    '''

