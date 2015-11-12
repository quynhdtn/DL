from liir.nlp.ml.nn.base.StackDenoisingAutoEncoderEx import StackDenoisingAutoEncoderEx
import theano as th
__author__ = 'quynhdo'

from sklearn.datasets import load_svmlight_file

import pickle



if __name__=="__main__":
    import sys

    train_file_path=sys.argv[1]
    dev_file_path=sys.argv[2]
    we_size= int( sys.argv[3] )
    mdl_name=sys.argv[4]

    X_train,y_train = load_svmlight_file(train_file_path)
    X_dev, y_dev = load_svmlight_file(dev_file_path)

    X_train_normal=X_train[:,0:X_train.shape[1]-1 - we_size]
    X_train_we=X_train[:,X_train.shape[1]-we_size:].toarray()


#X_train,y_train = load_svmlight_file("/Users/quynhdo/Documents/WORKING/PhD/NNSRL/tempDir/google_simple_data/pi_ood_N")

#X_dev, y_dev = load_svmlight_file("/Users/quynhdo/Documents/WORKING/PhD/NNSRL/tempDir/google_simple_data/pi_dev_N")
    print (y_dev.shape)
    X_train_normal=th.shared(X_train[:,0:X_train.shape[1]-1-we_size], borrow=True)
    X_train_we=th.shared(X_train[:,X_train.shape[1]-we_size:].toarray(), borrow=True)

    X_dev_normal=X_dev[:,0:X_dev.shape[1]-1-we_size]
    X_dev_we=X_dev[:,X_dev.shape[1]-we_size:].toarray()



    numHiddenSet=[[200,200,200],[250,250,250],[350,350,350],[400,400,400],[450,450,450],[500,500,500],[600,600,600],[700,700,700]]
    import theano.tensor as T

    pre_lr = [0.001,0.01]
    train_lr=[0.01,0.1]


    for ni in range(len(numHiddenSet)):
        numHidden=numHiddenSet[ni]

        for pl in pre_lr:
            for tl in train_lr:
                mdl_path=str(ni)+"_"+str(pre_lr.index(pl))+"_"+str(train_lr.index(tl))+"_"+mdl_name +".mdl"
                print (mdl_path)

                sda= StackDenoisingAutoEncoderEx(we_size,numHiddens=numHidden,numOutput=2, numExtend=238018, corruption_level=[0.1,0.2,0.3], extend_input_type="sparse")
                sda.preTraining(X_train_we, 20, 25,pl)
                sda.mlp.fit(X_train_we,X_train_normal, T.cast(th.shared(y_train, borrow=True), 'int32',
                                                                                                ), 20,1000,tl,
                            validation_data=(X_dev_we, X_dev_normal,y_dev), save_model_path=mdl_path)


            '''
                y_pred = sda.mlp.predict(X_dev_we, X_dev_normal)

                import pickle

                f =open("pi_model.pkl", "wb")
                pickle.dump(sda, f)
                f.close()


                print(y_pred)
                print (y_pred.shape)
                #print (test_set_y)

                from sklearn import metrics
                print (metrics.accuracy_score(y_dev,y_pred))
            '''

    '''
    import pickle

    f =open("pi_model.pkl", "rb")
    sda=pickle.load()

    X_dev, y_dev = load_svmlight_file("/Users/quynhdo/Documents/WORKING/PhD/NNSRL/tempDir/google_simple_data/pi_dev_N")
    print (y_dev.shape)

    X_dev_normal=X_dev[:,0:X_dev.shape[1]-301]
    X_dev_we=X_dev[:,X_dev.shape[1]-300:].toarray()

    y_pred = sda.mlp.predict(X_dev_we, X_dev_normal)

    from sklearn import metrics
    print (metrics.accuracy_score(y_dev,y_pred))

    f.close()
    '''
