import variables as var
import matrixOps as mOps
from math import e

def leaky_relu(raw, act):
    """Performs the Leaky ReLu activation function on a given layer.
    Stores the activated values into the list for the activated values.
    
    :param raw: the raw values of the layer computed from doing
        the dot product between the previous layer activated, or the
        inputs if that layer is the inputs, dotted by the weights of the
        current layer.
    :type raw: list of list of float
    :param act: the list to hold the output of the Leaky ReLu activation
        function when the raw value at the corresponding index has been
        passed in.
    :type act: list of list float
    """
    
    for r in range(len(act)):
        for c in range(len(act[0])):
            if raw[r][c] < 0 :
                act[r][c] = raw[r][c] * var.reluCoef
            else:
                act[r][c] = raw[r][c]
            
def softmax(raw, act):
    """Performs the Softmax activation function on a given layer. This
    activation function was meant to be used for the raw values of the
    last layer. Stores the activated values into the list for the
    activated values.
    
    :param raw: the raw values of the layer computed from doing
        the dot product between the previous layer activated, dotted by
        the weights of the current layer.
    :type raw: list of list of float
    :param act: the list to hold the output of the Leaky ReLu activation
        function when the raw value at the corresponding index has been
        passed in.
    :type act: list of list float
    """
    
    esum = 0
    # Raising e to the power of the elements in the list and find the sum
    for r in range(len(raw)):
        for c in range(len(raw[0])):
            esum += e**raw[r][c]
        
        # Finding the probabilities
        for c in range(len(raw[0])):
            act[r][c] = e**raw[r][c] / esum
        esum = 0


def makePrediction(inputs):
    """Makes a prediction using forward propagation from a given set
    of inputs. The inputs are an EMG recording of a gesture the user
    wishes the microcontroller to model.
    
    :param inputs: the EMG recording of a gesture made up of the
        highest average values recorded for each of the 2 EMG sensors
        while the gesture was performed. Values must be normalized by
        dividing by 65535, the max analog value.
    :type inputs: list of list of float
    
    :return: returns an 'int' type representing the gesture performed.
        0 means to perform a fist, 1 means to flex the index finger, 2
        means to flex both the index and middle finger, 3 means to flex
        the middle finger, and 4 means to flex the ring and pinky finger.
    """
    
    # Neural network with 2 Hidden Layers
    mOps.dotProduct(inputs, var.w1, var.layer1_raw)
    mOps.addBiases(var.layer1_raw, var.b1)
    leaky_relu(var.layer1_raw, var.layer1_act)

    mOps.dotProduct(var.layer1_act, var.w2, var.layer2_raw)
    mOps.addBiases(var.layer2_raw, var.b2)
    leaky_relu(var.layer2_raw, var.layer2_act)
    
    mOps.dotProduct(var.layer2_act, var.w3, var.layer3_raw)
    mOps.addBiases(var.layer3_raw, var.b3)
    softmax(var.layer3_raw, var.layer3_act)
    
    largestIdx = 0
    for i in range(len(var.layer3_act[0])):
        if var.layer3_act[0][i] > var.layer3_act[0][largestIdx]:
            largestIdx = i
    return largestIdx
        