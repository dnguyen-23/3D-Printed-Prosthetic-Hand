import variables as var
import matrixOps as mOps
from math import e

def leaky_relu(raw, act):
    for r in range(len(act)):
        for c in range(len(act[0])):
            if raw[r][c] < 0 :
                act[r][c] = raw[r][c] * var.reluCoef
            else:
                act[r][c] = raw[r][c]
    # Do not need to return layer
    # Operation modifies reference not just value
            
def softmax(raw, act):
    esum = 0
    # Raising e to the power of the elements in the list and find the sum
    for r in range(len(raw)):
        for c in range(len(raw[0])):
            esum += e**raw[r][c]
        
        # Finding the probabilities
        for c in range(len(raw[0])):
            act[r][c] = e**raw[r][c] / esum
        esum = 0

# make sure that "inputs" is a 2d array
def makePrediction(inputs):
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
        