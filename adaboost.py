#!usr/bin/python3

# Last Update : December 10, 2017
# Author : Dhaval R Niphade
# Course : CSCI B551 Elements of Artificial Intelligence
# Assignment 4 - Question 2

import random, math
import numpy as np
from json import load, dump
from copy import deepcopy

class AdaBoost():

    def __init__(self,trainFile=None,testFile=None):
        self.adaboost = {}
        if trainFile is not None:
            self.trainFile = self.loadModel(trainFile)
        if testFile is not None:
            self.testFile = self.loadModel(testFile)
        self.allStumps = {}

    def loadModel(self,modelFile):
        files = {}
        input_file = open(modelFile, 'r')
        for line in input_file:
            data = line.split()
            img = np.empty(192, dtype=np.int)
            index = 2
            i = 0
            while i < 192:
                img[i] = int(data[index])
                index += 1
                i += 1
            files[data[0] + data[1]] = {"orient": int(data[1]), "img": img}
        input_file.close()
        return files

    def setWeights(self,train,total):
        for example in train:
            train[example]["weight"] = 1.0 / total

    def getBest(self,boost,imageOrient):
        for trainFileID in self.trainFile:
            for pixels in boost:
                p = [int(pixel) for pixel in pixels.split()]
                if(self.trainFile[trainFileID]["img"][p[0]] > self.trainFile[trainFileID]["img"][p[1]])\
                    and (self.trainFile[trainFileID]["orient"] == imageOrient)\
                    or (self.trainFile[trainFileID]["img"][p[0]] < self.trainFile[trainFileID]["img"][p[1]]) \
                    and (self.trainFile[trainFileID]["orient"] != imageOrient):
                    boost[pixels]["value"] += self.trainFile[trainFileID]["weight"]

        maxPixel = max([[pixel, boost[pixel]] for pixel in boost], key=lambda x: x[1]["value"])
        return maxPixel

    def modifyWeight(self,modifier,stumpPix,imageOrient):
        nSum=0
        p = [int(pixel) for pixel in stumpPix.split()]
        for id in self.trainFile:
            if (self.trainFile[id]["img"][p[0]] > self.trainFile[id]["img"][p[1]]) and (self.trainFile[id]["orient"] == imageOrient) or (self.trainFile[id]["img"][p[0]] < self.trainFile[id]["img"][p[1]]) and (self.trainFile[id]["orient"]!= imageOrient):
                self.trainFile[id]["weight"] *= modifier
            nSum+=self.trainFile[id]["weight"]
        return nSum

    def normalize(self,value):
        for id in self.trainFile:
            self.trainFile[id]["weight"] = self.trainFile[id]["weight"] / value

    def train(self,fileName):

        # if not self.trainFile:
        self.trainFile = self.loadModel(fileName)

        numTrain = len(self.trainFile)
        stumpCount = 50
        print("Done loading files")

        for i in range(stumpCount):
            p1,p2 = -1,-1
            while p1==-1 or p2==-1 or (str(p1)+" "+str(p2)) in self.adaboost:
                p1,p2 = random.randint(0,191),random.randint(0,191)
            self.adaboost[str(p1)+" "+str(p2)] = {'value':0}

        print("Done generating stumps")

        for orient in [0,90,180,270]:
            bestAttr = []
            self.setWeights(self.trainFile,numTrain)
            # for vector in self.trainFile:
            #     self.trainFile[vector]["weight"] = 1.0/numTrain
            print("Weights set for ", orient)
            newBoost = deepcopy(self.adaboost)
            for stump in range(stumpCount):
                bestAttr.append(self.getBest(newBoost,orient))
                totalWeight = sum([self.trainFile[train]["weight"] for train in self.trainFile])
                error = (totalWeight - bestAttr[stump][1]["value"]) / totalWeight

                # Preventing division by 0 error
                error = 0.99 if error == 1 else error
                beta = error /(1 - error)

                bestAttr[stump].append(1 + math.log(1/beta))
                normalizeSum = self.modifyWeight(beta,bestAttr[stump][0], orient)
                self.normalize(normalizeSum)
                del newBoost[bestAttr[stump][0]]

                for key in newBoost:
                    newBoost[key]["value"] = 0

            self.allStumps[orient] = bestAttr

        print("Done training")


    def classify(self,filename,modelFile):

        self.testFile = self.loadModel(filename)

        countCorrect=0
        confusionMatrix = np.zeros((4,4),dtype=np.int)
        for id in self.testFile:
            finLabel = {}
            for orient in self.allStumps:
                decision = 0
                for decStump in self.allStumps[orient]:
                    p = [int(pixel) for pixel in decStump[0].split()]
                    if self.testFile[id]["img"][p[0]] > self.testFile[id]["img"][p[1]]:
                        decision += decStump[2] * 1
                    else:
                        decision += decStump[2] * (-1)
                finLabel[orient] = decision
            decOrient = max([[key,finLabel[key]] for key in finLabel], key=lambda x : x[1])
            if self.testFile[id]["orient"] == decOrient[0]:
                countCorrect+=1
            confusionMatrix[self.testFile[id]["orient"]/90, (decOrient[0])/90] += 1

        print("Confusion Matrix = \n", str(confusionMatrix))
        print("Accuracy: ", (countCorrect * 100)/len(self.testFile))

# def main():
#
#     mode, targetFile, modelFile, model = sys.argv[1:]
#     ada = AdaBoost("train-data.txt","test-data.txt")
#
#     if mode == "train":
#         ada.train(targetFile)
#     else:
#         ada.classify(targetFile)
#
# if __name__ == '__main__':
#     main()