__author__ = 'quynhdo'

from ml.liblinear.liblinearutil import *

y, x = svm_read_problem('train.forsvm.txt')

m = train(y, x, '-c 4')