__author__ = 'quynhdo'

# this file is used to load data from feature data files

import os
import numpy as np
import operator
import pickle
import scipy.sparse
class DataLoader:

    def __init__(self, feature_path):
        '''

        :param path:  Path to the folder that store all the feature vobcabulary files
        :return:
        '''


        self.features={}  # the dictionary contains the dimension of each features: feature_name:dim

        files = []
        for name in os.listdir(feature_path):
            fn = os.path.join(feature_path, name)
            if (fn.endswith("_vob.txt")):
                if os.path.isfile(fn):
                    files.append(os.path.join(feature_path, name))


        for fn in files:
            f = open (fn, "rb")
            name = os.path.basename(f.name).split("_")[0]
            lines=[l.strip() for l in f.readlines() if l.strip()!= ""]
            self.features[name]=len(lines)
            f.close()

    def getFeatureDim(self):
        total = 0
        for i in self.features.values():
            total+=i
        return total

    def loadDataAsSparse(self, path):
        labels = []

        f = open (os.path.join(path,"data.txt"))    # read all instance labels
        for line in f.readlines():
            line = line.strip()
            if line != "":

                l = line.split("\t")[2]
                if l == "Yes": l = "1"
                else: l = "-1"
                labels.append(l)
        f.close()
        print ("Reading %d instances..." %len(labels))


        data = scipy.sparse.lil_matrix((len(labels), self.getFeatureDim()))

        f_idx = 0
        for fn in self.features.keys():  # iterate through the features
            f = open (os.path.join(path, fn+ ".txt"))
         #   print (os.path.join(path, fn+ ".txt"))
            ins_idx=0
            for line in f.readlines():
                line = line.strip()
                if line!="":
                 #   print (line)
                    tmps = line.split("\t")

                    for tmp in tmps:
              #          print (tmp)
                        tm = tmp.split ( ":")
              #          print(ins_idx)
              #          print(tm)
              #          print(f_idx)
                        if (len(tm) > 1):
                            data[ins_idx,f_idx + int(tm[0])] = int(tm[1])
                    ins_idx+=1
            f.close()

            f_idx+=self.features[fn]

#        print (ins_idx)
#        print (f_idx)

        return (data, np.asarray(labels))

    def loadData(self, path):
        '''
        Load data features
        :param path: path to folder contain all data file
        :return:
        '''
        labels = []

        f = open (os.path.join(path,"data.txt"))    # read all instance labels
        for line in f.readlines():
            line = line.strip()
            if line != "":

                l = line.split("\t")[2]
                if l == "Yes": l = "1"
                else: l = "-1"
                labels.append(l)
        f.close()
        print ("Reading %d instances..." %len(labels))


        data = np.zeros((len(labels), self.getFeatureDim()), "int32")

        f_idx = 0
        print (self.features.keys())
        for fn in self.features.keys():  # iterate through the features
            f = open (os.path.join(path, fn+ ".txt"))
         #   print (os.path.join(path, fn+ ".txt"))
            ins_idx=0
            for line in f.readlines():
                line = line.strip()
                if line!="":
                 #   print (line)
                    tmps = line.split("\t")

                    for tmp in tmps:
              #          print (tmp)
                        tm = tmp.split ( ":")
              #          print(ins_idx)
              #          print(tm)
              #          print(f_idx)
                        if (len(tm) > 1):
                            data[ins_idx][f_idx + int(tm[0])] = int(tm[1])
                    ins_idx+=1
            f.close()

            f_idx+=self.features[fn]

#        print (ins_idx)
#        print (f_idx)

        return (data, np.asarray(labels))


    def loadDataAndWriteToFile(self, pathIn, pathOut):
        '''
        Load data features
        :param path: path to folder contain all data file
        :return:
        '''
        labels = []

        f = open (os.path.join(pathIn,"data.txt"))    # read all instance labels
        for line in f.readlines():
            line = line.strip()
            if line != "":
                l = line.split("\t")[2]
                if l == "Yes": l = "1"
                else: l = "-1"
                labels.append(l)
        f.close()
        print ("Reading %d instances..." %len(labels))




        f_idx = 0
        for fn in self.features.keys():  # iterate through the features
            f = open (os.path.join(pathIn, fn+ ".txt"))
         #   print (os.path.join(path, fn+ ".txt"))
            ins_idx=0
            for line in f.readlines():
                line = line.strip()
                if line!="":
                 #   print (line)

                    tmps = line.split("\t")

                    d= {}
                    for tmp in tmps:
              #          print (tmp)
                        tm = tmp.split ( ":")
              #          print(ins_idx)
              #          print(tm)
              #          print(f_idx)
                        if (len(tm) > 1):
                            d[f_idx + int(tm[0])] = tm[1]
                    t=sorted(d.items(), key=operator.itemgetter(0),reverse=False)
                    for k in range(len(t)):
                            labels[ins_idx]+="\t"+ str(t[k][0]) +":" + t[k][1]

                    ins_idx+=1
            f.close()

            f_idx+=self.features[fn]

        fout = open (pathOut, "w")

        for l in labels:
            fout.write(l + "\n")

#        print (ins_idx)
#        print (f_idx)

        fout.close()



    def writeToFile(self, dataAndlabel, path):
        f = open (path,  "w")
        datas,labels =  dataAndlabel
        for idx in range(len(labels)):
            f.write(str(labels[idx]) + "\t")
            for f_idx in range(len(datas[0])):
                if datas[idx][f_idx]!=0:
                    f.write(str(f_idx)+":"+str(datas[idx][f_idx])+ "\t")

            f.write("\n")

        f.close()




dl = DataLoader("/Users/quynhdo/Documents/WORKING/PhD/workspace/WE/NNSRL/train_v/fea")
'''
print(dl.getFeatureDim())

train_data = dl.loadDataAsSparse("/Users/quynhdo/Documents/WORKING/PhD/workspace/WE/NNSRL/train_v")
from sklearn import svm, metrics
clf =  svm.SVC(kernel='linear', C = 1.0)

print ( "Start training...")
clf.fit(train_data[0],train_data[1])
f =open("pi_model.pkl", "wb")
pickle.dump(clf,f)
f.close()

'''


#print(dl.features)
#print(dl.getFeatureDim())

train_data = dl.loadData("/Users/quynhdo/Documents/WORKING/PhD/workspace/WE/NNSRL/eval_v")
#dl.loadDataAndWriteToFile("/Users/quynhdo/Documents/WORKING/PhD/workspace/WE/NNSRL/eval_v", "eval_v_pi.forsvm.txt")


#ins_test=dl.loadData("/Users/quynhdo/Documents/WORKING/PhD/workspace/WE/NNSRL/eval")

#from sklearn import svm, metrics

#clf =  svm.SVC(kernel='linear', C = 1.0)
#print ("start training...")

#clf.fit(data, labels)
#f =open("pi_model.pkl", "rb")
#clf = pickle.load(f)
#f.close()
#print (ins_test[0])
#y_pred = clf.predict(ins_test[0])
#print ("label..")
#print (y_pred)

#print (ins_test[1])

#f =open("pi_model.pkl", "wb")
#pickle.dump(clf, f)
#f.close()

#print (metrics.f1_score(ins_test[1],y_pred))

from sklearn.datasets import load_svmlight_file
#X_train, y_train = load_svmlight_file("train_v_pi.forsvm.txt")
#X_test, y_test=dl.loadData("/Users/quynhdo/Documents/WORKING/PhD/workspace/WE/NNSRL/eval_v")



#from sklearn import svm, metrics
#clf =  svm.SVC(kernel='linear', C = 1.0)

#print ("Start training...")
#clf.fit(X_train,y_train)

#f =open("pi_model.pkl", "wb")
#pickle.dump(clf,f)
#f.close()
#f =open("pi_model.pkl", "rb")
#clf = pickle.load(f)
#f.close()
#print (ins_test[0])
#y_pred = clf.predict(X_test)
#print ("label..")
#print (y_pred)

#print (y_test)

#print (ins_test[1])

#f =open("pi_model.pkl", "wb")
#pickle.dump(clf, f)
#f.close()

#print (metrics.accuracy_score(y_test,y_pred))

