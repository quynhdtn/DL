__author__ = 'quynhdo'
import pickle, gzip, numpy


with open('/Users/quynhdo/Downloads/mnist.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    print(p)