__author__ = 'quynhdo'

#   Type =  Enum(['FEED_FORWARD',"RECURRENT",'CONVOLUTIONAL','SUBSAMPLING','RECURSIVE','MULTILAYER'])
#   TrainingMode = Enum(['TRAIN', 'TEST'])

class Enum(tuple): __getattr__ = tuple.index