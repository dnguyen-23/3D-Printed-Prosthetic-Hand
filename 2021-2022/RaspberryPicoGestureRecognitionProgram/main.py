from machine import ADC, Timer, Pin
import forwardPass as fP
import time
import random
import variables as var
import backwardsPass as bP
import matrixOps as mOps
IMemg = ADC(26)
RPemg = ADC(27)

Button = Pin(14, Pin.IN, Pin.PULL_DOWN)

IMlist = [0, 0, 0, 0, 0]
RPlist = [0, 0, 0, 0, 0]
idx = 0
recorded =0
thresholdVal = 8000
activating = False
IMmax = 0
RPmax = 0



def readEMG(thresholdVal, IMemg, RPemg, predict):
 #taking the average of the values
    global idx
    global IMlist
    global IMmax
    
    global RPlist
    global RPmax
    global activating
   
    IMlist[idx] = IMemg.read_u16()
    RPlist[idx] = RPemg.read_u16()
    idx += 1
    if idx == 5:
        idx = 0

    IMsum = 0
    RPsum = 0
    for reading in range(len(IMlist)):
        IMsum += IMlist[reading]
        RPsum += RPlist[reading]
        
    avgIM = IMsum / len(IMlist)
    avgRP = RPsum / len(RPlist)
  
    if avgIM > IMmax:
        IMmax = avgIM
    
    if avgRP > RPmax:
        RPmax = avgRP
    print(avgIM, avgRP)
    
    if (avgIM > thresholdVal or avgRP > thresholdVal) and not activating:
        activating = True  
    
    if (avgIM < thresholdVal and avgRP < thresholdVal) and activating:
        activating = False
        readings = [[IMmax / var.normalizeFactor, RPmax / var.normalizeFactor]]
        IMmax = 0
        RPmax = 0
        if predict == True:
            gestures = {0 : "flex fist",
                        1 : "flex index",
                        2 : "flex index and middle",
                        3 : "flex middle",
                        4 : "flex ring and pinky"}
            print(gestures[fP.makePrediction(readings)])
        else:
            return readings
        time.sleep(1)
#     gestureIdx = makePrediction(IMmax, RPmax)
#         recordToCSV(IMmax, RPmax, "flex index middle", gestureFile)


def recordToCSV(IM, RP, gesture, file):
    string = str(IM) + "," + str(RP) + "," + gesture + "\n"
    file.write(string)
    global recorded
    recorded += 1
    print("recorded: ", recorded)
    time.sleep(1.5)

def optimize(numEpochs):
    #1.) Obtain the training data from the user
    data = []
    labels = []
    for g in range(var.numGestures):
        for n in range(10):
            data.append(readEMG(thresholdVal, IMemg, RPemg, False))
            labels.append(g)
    
    #2.) Individually feed each training data into the backProp function and shuffle after every cycle
    for n in range(numEpochs):
        shuffle(data, labels, len(labels))
        for i in range(len(data)):
            inputs = [data[i]]
            bP.backProp(inputs, labels[i])
        
#     for x in range(numEpochs):
#         for i in range(var.numSamples):
#             bP.backProp([var.trainingData[i]], var.trainingLabels[i])
#             shuffle(var.trainingData, var.trainingLabels, var.numSamples)
#         if x % 5 == 0:
#             print(testAccuracy())
    print("Finished optimization. Final accuracy: ", testAccuracy() * 100)

def testAccuracy():
    accuracy = 0
    for i in range(var.numSamples):
        p = fP.makePrediction([var.testingData[i]])
        if p == var.testingLabels[i]:
            accuracy += 1
    return accuracy / var.numSamples
# while True:
#     #run the neural network, taking inputs and make a prediction of what the desired gesture was
#     # check if switch was pressed, if so, then optimize the neural network
#     if Button.vamainlue() == 1:
#         optimize()

def shuffle(data, labels, length):
    for i in range(int(length / 2)):
        randIdx = random.randint(i, length - 1)
        tempData = data[randIdx]
        tempLabel = labels[randIdx]
        
        data[randIdx] = data[i]
        labels[randIdx] = labels[i]
        
        
        data[i] = tempData
        labels[i] = tempLabel


while True:
    readEMG(thresholdVal, IMemg, RPemg, True)
    time.sleep(0.2)
    if Button.value() == 1:
        optimize(10)
        time.sleep(0.5)
