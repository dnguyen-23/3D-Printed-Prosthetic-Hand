from machine import ADC, Timer, Pin, I2C, PWM
import utime
import forwardPass as fP
import time
import random
import variables as var
import backwardsPass as bP
import matrixOps as mOps

IMemg = ADC(26)
RPemg = ADC(27)

OptButton = Pin(14, Pin.IN, Pin.PULL_DOWN)
TestButton = Pin(15, Pin.IN, Pin.PULL_DOWN)

IMlist = [0, 0, 0, 0, 0]
RPlist = [0, 0, 0, 0, 0]
idx = 0
recorded =0
thresholdVal = 10000
activating = False
IMmax = 0
RPmax = 0

iServo = PWM(Pin(0))
mServo = PWM(Pin(1))
rpServo = PWM(Pin(2))

iServo.freq(50)
mServo.freq(50)
rpServo.freq(50)

iAct = False
mAct = False
rpAct = False

def readEMG(thresholdVal, IMemg, RPemg, predict):
    
    """Checks if the user is performing a gesture and actively records values
    from the EMG sensors if the user is performing a gesture. This function
    actively calculates the average of the previous 5 values for each EMG
    sensor. The highest average value recorded once the user has finished
    performing a gesture will be the values making up a recording for a
    gesture. If "predict" is true, then an integer representing a gesture
    will be returned. This is where the user of the prosthetic hand should
    implement the function to actuate the motors. Otherwise, the EMG
    recording will be returned.
    
    :param thresholdVal: value used check if the user is performing a gesture
    :type thresholdVal: int
    :param IMemg: GPIO pin of EMG sensor monitoring index and middle finger
    :type IMemg: ADC
    :param RPemg: GPIO pin of EMG sensor monitoring ring and pinky finger
    :type RPemg: ADC
    :param predict: decides if function activates motors or return EMG recordings
    :type predict: bool
    
    :return: the EMG recording of the gesture if predict is false or an
        integer representing a gesture if predict is true. 0 is a fist, 1
        is a flexed index finger, 2 is the index and middle finger flexed,
        3 is the middle finger flexed, and 4 is the ring and pinky finger flexed.
    """

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
            # This is where the motor should be activated. Instead of printing the integer
            # the integer should be used to actuate the motors on the prosthetic hand 
            predIdx = fP.makePrediction(readings)
            predictedGesture = (var.gestures[predIdx])
            print(predictedGesture)
            # print(" ")
            i2c = I2C(0, scl=Pin(17), sda=Pin(16), freq=100000)
            addr = i2c.scan()[0] #getting the address of the peripheral

            i2c.writeto(addr, str(predIdx))
            time.sleep(2)
            
        else:
            return readings
        #time.sleep(1)


def recordToCSV(IM, RP, gesture, file):
    
    """Records the EMG recordings of a gesture into a CSV file
    
    :param IM: highest average value of EMG sensor monitoring index/middle finger
    :type IM: float
    :param RP: highest average value of EMG sensor monitoring ring/pinky finger
    :type RP: float
    :param gesture: the label for the EMG recording
    :type gesture: str
    :param file: the path of the CSV file the recording is written to
    :type file: str:
    """
    
    string = str(IM) + "," + str(RP) + "," + gesture + "\n"
    file.write(string)
    global recorded
    recorded += 1
    print("recorded: ", recorded)
    time.sleep(1.5)

def optimize(numEpochs, testing):
    
    """Optimizes the neural network
    
    :param numEpochs: the number of epochs to train the neural network
    :type numEpochs: int
    :param testing: determines if program is being demonstrated or in real world use
    :type testing: bool
    """
    #1.) Obtain the training data from the user
    data = []
    labels = []
    
    if testing == False:
        for g in range(var.numGestures):
            for n in range(10):
                data.append(readEMG(thresholdVal, IMemg, RPemg, False))
                labels.append(g)
    elif testing == True: #demonstrating
        data = var.trainingData
        labels = var.trainingLabels
    length = len(labels)
    
    #2.) Individually feed each training data into the backProp function and shuffle after every cycle
    for n in range(numEpochs):
        if n % 10 == 0:
            print(testAccuracy())
            
        shuffle(data, labels)

        for i in range(length):
            inputs = [data[i]]
            bP.backProp(inputs, labels[i])
        
    print("Finished optimization. Final accuracy: ", testAccuracy() * 100, "%")



def testAccuracy():
   
    """Tests the accuracy of the neural network model"""
    
    accuracy = 0
    for i in range(var.numSamples):
        p = fP.makePrediction([var.testingData[i]])
        if p == var.testingLabels[i]:
            accuracy += 1
    return accuracy / var.numSamples

def shuffle(data, labels):
    
    """Shuffles a given set of data and its labels
    :param data: a list of EMG recordings of gestures
    
    :type data: list of list of float
    :param labels: a list of the labels corresponding to data
    :type labels: list of int
    """
    length = len(data)
    for x in range(20):
        for i in range(length - 1):
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
    if OptButton.value() == 1:
#         optimize(30, False) # real world use
        optimize(30, True) # demonstration
        time.sleep(0.5)
    if TestButton.value() == 1:
        print("Accuracy: ", 100 * testAccuracy(), "%")
        time.sleep(0.5)


