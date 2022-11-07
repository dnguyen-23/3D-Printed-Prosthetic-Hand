# This is a sample Python script.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Press Ctrl+Shift+. to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Data from the Arduino
# dataIdx = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles/100_FlexIndex.csv")
# dataMid = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles/100_FlexMid.csv")
# dataIdxMid = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles/100_FlexIndexAndMid.csv")
# dataRingPinky = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles/100_FlexRingPinky.csv")
# dataFist = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles/100_FlexFist.csv")

# Data from the raspberry pico
dataIdx = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexIdx.csv")
dataMid = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexMid.csv")
dataIdxMid = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexIM.csv")
dataRingPinky = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexRP.csv")
dataFist = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexFist.csv")
print(dataIdx.head())

dataIdx = np.array(dataIdx)
dataMid = np.array(dataMid)
dataRingPinky = np.array(dataRingPinky)
dataFist = np.array(dataFist)
dataIdxMid = np.array(dataIdxMid)
listOfColors = ['r*', 'b*', 'g*', 'y*', 'm*']

listOfData = [dataIdx, dataMid, dataRingPinky, dataFist, dataIdxMid]
plt.title("Recordings for 5 Gestures Recorded from Raspberry Pico")
plt.xlabel("EMG Sensor Readings for Index and Middle Finger")
plt.ylabel("EMG Sensor Readings for Ring and Pinky Finger")
plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams["figure.autolayout"] = True
# plt.yscale('log')
# plt.xscale('log')
gestures = ["Index", "Middle", "Ring and Pinky", "Fist", "Index and Middle"]
plt.legend(loc = 0)
for i in range(5):
    data = listOfData[i]
    idxMid = data.T[0] + 1
    ringPinky = data.T[1] + 1

indexFinger, = plt.plot(listOfData[0].T[0] + 1, listOfData[0].T[1] + 1, listOfColors[0])
middleFinger, = plt.plot(listOfData[1].T[0] + 1, listOfData[1].T[1] + 1, listOfColors[1])
rPFinger, = plt.plot(listOfData[2].T[0] + 1, listOfData[2].T[1], listOfColors[2])
flexFist, = plt.plot(listOfData[3].T[0] + 1, listOfData[3].T[1], listOfColors[3])
iMFinger, = plt.plot(listOfData[4].T[0] + 1, listOfData[4].T[1], listOfColors[4])

plt.legend(handles = [indexFinger, middleFinger, rPFinger, flexFist, iMFinger],
           labels = gestures)
plt.axis([1, 45000, 1, 65000]) # this if for plotting the raspberry pico
# plt.axis([1, 1023, -10, 1023])

plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
