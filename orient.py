#!usr/bin/python3

# Last Update : December 9, 2017
# Author : Dhaval R Niphade
# Course : CSCI B551 Elements of Artificial Intelligence
# Assignment 4 - Question 1

import sys, shutil, pickle
from collections import defaultdict
import adaboost
import numpy as np

class KNN():

    K = 7
    def __init__(self):
        pass

    def train(self,filename):
        shutil.copyfile(filename,"model.txt")
        return

    def writeToFile(self,id,label):
        with open("output.txt","a+") as f:
            f.write(' '.join(map(str,[id,label])))
            f.write("\n")


    def loadModel(self,modelFile):
        allVectors = np.empty((1,192))
        labels, ids = [], []
        with open(modelFile,'r') as f:
            lines = f.readlines()
        for i,line in enumerate(lines):
            tokens = line.strip().split()
            if tokens:
                ids.append(tokens[0])
                labels.append(int(tokens[1]))
                x = [int(x) for x in tokens[2:]]
            if i==0:
                allVectors = np.array(x).reshape((1,192))
            else:
                allVectors = np.concatenate((allVectors,np.array(x).reshape((1,192))),axis=0)
        print(allVectors.shape)
        return allVectors,labels,ids

    def classify(self,testFile,modelFile):

        trainModel,trainLabels,ids = self.loadModel(modelFile)
        testSet,testLabels,ids = self.loadModel(testFile)

        x_train,y_train = trainModel.shape
        x_test,y_test = testSet.shape
        ones = np.ones((x_train, 1))
        god = []

        for i in range(x_test):
            test = np.dot(ones,np.array(testSet[i]).reshape((1,192)))

            A = trainModel - test
            A = np.dot(A,np.transpose(A))
            final = np.diag(A)
            final = list(final)
            results = sorted(final)

            # Assemble K neighbours
            predictions = defaultdict(int)
            for j in range(KNN.K):
                posInFinal = final.index(results[j])
                predictions[trainLabels[posInFinal]]+=1

            # Voting for top
            currMax,ansLabel = -1, ""
            for k,v in predictions.items():
                if v > currMax:
                    currMax = v
                    ansLabel = k

            # print("Predicted Output = ", ansLabel , " Expected output = ", testLabels[i])
            god.append(ansLabel == testLabels[i])
            self.writeToFile(ids[i],ansLabel)

        print("Accuracy = ", float(sum(god)/len(god)))


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():

    mode, filename, modelFile, model = sys.argv[1:]

    # Driver
    if mode == "train":
        if model == "nearest" or model == "best":
            knn = KNN()
            knn.train(filename)
        elif model == "adaboost":
            ada = adaboost.AdaBoost()
            ada.train(filename)
            save_obj(ada,modelFile)
        elif model == "nnet":
            pass
        else:
            print("Incorrect model.........exiting")
            sys.exit(1)

    elif mode == "test":
        if model == "nearest" or model == "best":
            knn = KNN()
            knn.classify(filename,modelFile)
        elif model == "adaboost":
            ada = load_obj(modelFile)
            ada.classify(filename,modelFile)
        elif model == "nnet":
            pass
        else:
            print("Incorrect model....exiting")
            sys.exit(1)

    else:
        print("Incorrect mode......exiting")
        sys.exit(1)

if __name__ == '__main__':
    main()


# REDUNDANT BLOCKS OF CODE - NO LONGER REQUIRED

# def getEuclid(self, vec1, vec2):
#
#     vec1 = list(map(int, vec1))
#     vec2 = list(map(int, vec2))
#
#     if len(vec1) != len(vec2):
#         print("Incomparable vectors")
#         sys.exit(1)
#
#     dist = [(a-b)**2 for a,b in zip(vec1,vec2)]
#     dist = math.sqrt(sum(dist))
#     return dist
