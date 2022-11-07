import tensorflow as tf
import math
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
LE = preprocessing.LabelEncoder() #this is an object that turns non-numerical data to numerical data

dataIdx = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexIdx.csv")
dataMid = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexMid.csv")
dataIdxMid = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexIM.csv")
dataRingPinky = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexRP.csv")
dataFist = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexFist.csv")

allData = [dataIdx, dataMid, dataRingPinky, dataFist, dataIdxMid]
numTraining = 160
numTesting = 10

allDataTrain = [dataIdx[0: numTraining], dataMid[0:numTraining], dataRingPinky[0:numTraining], dataFist[0:numTraining], dataIdxMid[0:numTraining]]
allDataTest = [dataIdx[numTraining:numTraining + numTesting], dataMid[numTraining:numTraining + numTesting],
               dataRingPinky[numTraining:numTraining + numTesting], dataFist[numTraining:numTraining + numTesting],
               dataIdxMid[numTraining:numTraining + numTesting]]

x_dataxTrain = [] # holds the input values from the index and middle finger sensor
x_datayTrain = [] # holds the input values from the ring/pinky fingers sensor
y_dataTrain = [] # holds the labels

x_dataxTest = [] # holds the input values from the index and middle finger sensor
x_datayTest = [] # holds the input values from the ring/pinky fingers sensor
y_dataTest = [] # holds the labels


def separateData(data):
    """ Takes in a dataset of EMG recordings and their labels and separates
    the recordings from the labels.

    :param data: the 2D array holding the EMG recordings with their labels
    :type data: list of list of float
    :return: X represents the EMG recordings and Y represents the labels
    """
    copyData = []
    for i in range(len(data)):
        input = [1, 2, 3]
        input[0] = data[i][0]
        input[1] = data[i][1]
        input[2] = data[i][2]
        copyData.append(input)
    # print(copyData)
    copyData = np.array(copyData)
    copyData = copyData.T
    Xx = copyData[0]
    Xy = copyData[1]
    X_diff = Xx - Xy
    Y = copyData[2]
    X = np.array(list(zip(Xx, Xy, X_diff)))

    # print(X)
    # print(Y)
    return X, Y

# This method is for directly adding  on to the list by element
# this is different from appending another list onto the base list
def appendList(baseList, originalList):
    """Takes an EMG recording and its label, all kept together as a list of
    3 elements, and adds it as an element to the overall baselist
    :param baseList: a list representing either the training or testing data
    :type baseList: list of list
    :param originalList: represents the original list of the data
    :type originalList: list of list
    :return: the baselist with the data from the original list appended
    """
    for elem in originalList:
        baseList.append(elem)
    return baseList

def convertToVal(arr, hashmap):
    """Takes an array/list and converts the elements of the list
    to their corresponding value in the hashmap/dictionary

    :param arr: the original list of the elements to be converted
    :type arr: list of float
    :param hashmap: represents the map of keys and values. The
        elements of arr are keys for the values in hashmap
    :type hashmap: dict
    :return: the list with the elements converted to their
        corresponding values in the hashmap
    """
    for i in range(len(arr)):
        arr[i] = hashmap[arr[i]]
    return arr


for data in allDataTrain:
    appendList(x_dataxTrain, data['Index Middle Finger'])
    appendList(x_datayTrain, data['Ring Pinky Finger'])
    appendList(y_dataTrain, data['Gesture'])

for data in allDataTest:
    appendList(x_dataxTest, data['Index Middle Finger'])
    appendList(x_datayTest, data['Ring Pinky Finger'])
    appendList(y_dataTest, data['Gesture'])


pdY = pd.Series(y_dataTrain)
y_data_vocab = pdY.unique() #takes specifically the unique labels
numerical_y_train = LE.fit_transform(y_data_vocab) #these labels are then translated into numerical values
map = zip(y_data_vocab, numerical_y_train)
map = dict(map)

y_dataTrain = convertToVal(y_dataTrain, map)
y_dataTest = convertToVal(y_dataTest, map)

sortedDataTrain = list(zip(x_dataxTrain, x_datayTrain, y_dataTrain))
sortedDataTest = list(zip(x_dataxTest, x_datayTest, y_dataTest))

random.shuffle(sortedDataTrain)
random.shuffle(sortedDataTest)

x_train, y_train = separateData(sortedDataTrain)
x_test, y_test = separateData(sortedDataTest)


print(x_train)
maxValue = 65535
x_train = x_train / maxValue
x_test = x_test / maxValue