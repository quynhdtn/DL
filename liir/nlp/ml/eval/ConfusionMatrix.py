__author__ = 'quynhdo'
import numpy as np
class ConfusionMatrix:
    def __init__(self, n_classes, class_names = None):
        self.n_classes = n_classes
        self.class_names = class_names
        self.mat = np.zeros((n_classes, n_classes), dtype='int')

    def addBatch(self, y_true, y_predicted):
        assert  len(y_true) == len(y_predicted)
        for i in range(len(y_true)):

            self.mat[y_true[i],y_predicted[i]] +=1

    def __str__(self):
        s = "\t"
        for idx in range(self.n_classes): s += str(idx) + "\t"
        s += "\n"


        for i in range (len(self.mat)):
            s += str(i) + "\t"
            for j in range(len(self.mat[i])):
                s += str(self.mat[i][j]) + "\t"
            s += "\n"

        return s



    def getScore(self):
        num_instances = np.sum(self.mat, axis=1)
        predict = np.sum(self.mat, axis=0)

        correct = np.diag(self.mat).flatten()

        p = correct / predict * 100
        r = correct / num_instances * 100

        f = np.zeros (len(p))

        for i in range (len(p)):
            if (p[i]+ r[i] != 0):
                f = 2 * p * r / (p+r)
            else:
                f = None



        return  np.matrix([p, r,f]).transpose()






if __name__ == "__main__":

    cm= ConfusionMatrix(3)

    cm.addBatch([1,2,1,0],[2,2,0,0])

    print (cm.__str__())
    print (cm.getScore())


