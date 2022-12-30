import variables as var
import matrixOps as mOps
import forwardPass as fP
def dRelu(layer_raw, deriv):
    """Takes the derivative of the Sparse Categorical Cross Entropy
    loss function with respect to derivative of the Leaky ReLu
    activated values of a layer.
    
    :param layer_raw: the raw values of the layer that the Leaky
        ReLu activation function was performed on to obtain the
        activated values.
    :type layer_raw: list of list of float
    :param deriv: holds the derivatives of the Sparse Categorical
        Cross Entropy loss function with respect to the derivative
        of each Leaky ReLu activated value of the layer that layer_raw
        belongs to
    :type deriv: list of list of float
    """
    for i in range(len(layer_raw)):
        for j in range(len(layer_raw[0])):
            if layer_raw[i][j] <= 0.0:
                deriv[i][j] *= var.reluCoef
            elif layer_raw[i][j] > 0.0:
                deriv[i][j] *= 1
    
def backProp(inputs, labelIdx):
    """Takes a set of inputs as training data and optimizes the neural network
    through Gradient Descent, using the training data and its label. The neural
    network is optimized by taking the derivative of the Sparse Categorical
    Cross Entropy loss function with respect to each parameter/weight and bias
    in the neural network. The gradient value/derivative found will then be
    applied to the corresponding weight or bias after being multiplied by a
    learning rate.
    
    :param inputs: the EMG recording of a gesture made up of the
        highest average values recorded for each of the 2 EMG sensors
        while the gesture was performed. Values must be normalized by
        dividing by 65535, the max analog value.
    :type inputs: list of list of float
    :param labelIdx: the label for what the gesture performed was. 0 is a fist, 1 is
        the index finger flexed, 2 is the index and middle finger flexed, 3 is the
        middle finger flexed, and 4 is the ring and pinky finger flexed.
    :type labelIdx: int
    """
    #This function needs to run in order to get raw and activation values for each layer
    fP.makePrediction(inputs)
    
    dL3raw = var.layer3_act
    dL2raw = [[0] * var.l2Neurons] 
    dL1raw = [[0] * var.l1Neurons]
    
    dW3 = [[0 for g in range(var.numGestures)] for y in range(var.l2Neurons)]
    dW2 = [[0 for x in range(var.l2Neurons)] for y in range(var.l1Neurons)]
    dW1 = [[0 for x in range(var.l1Neurons)] for y in range(2)]
    
    # Getting derivative of loss with respect to derivative of the raw third layer (last layer)
    dL3raw[0][labelIdx] -= 1
    mOps.dotProduct(mOps.transpose(var.layer2_act), dL3raw, dW3)
    mOps.calcDBias(dL3raw, var.b3)
    
    # Getting derivative of loss with respect to derivative of the raw second layer
    mOps.dotProduct(dL3raw, mOps.transpose(var.w3), dL2raw)
    dRelu(var.layer2_raw, dL2raw)
    
    mOps.dotProduct(mOps.transpose(var.layer1_act), dL2raw, dW2)
    mOps.calcDBias(dL2raw, var.b2)
    
    
    # Getting derivative of loss with respect to derivative of the raw first layer
    mOps.dotProduct(dL2raw, mOps.transpose(var.w2), dL1raw)
    dRelu(var.layer1_raw, dL1raw)
    
    mOps.dotProduct(mOps.transpose(inputs), dL1raw, dW1)
    mOps.calcDBias(dL1raw, var.b1)
    
    mOps.applyDWeights(dW1, var.w1)
    mOps.applyDWeights(dW2, var.w2)
    mOps.applyDWeights(dW3, var.w3)
