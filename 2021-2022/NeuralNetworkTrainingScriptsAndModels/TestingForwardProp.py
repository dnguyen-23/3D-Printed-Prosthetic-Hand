from preparation import *

#Incomplete
w1 = np.array([[-0.7517679 ,  0.90327704, -0.2725665 ,  0.44247767],
       [-0.79656595, -0.898392  , -0.11328524,  0.31190982]])
w2 = np.array([[-0.24556546,  0.7765486 , -0.7371209 ,  0.30388987],
       [ 0.51157296,  0.8074439 ,  0.8614213 ,  0.28633314],
       [-0.49447647,  0.05812752,  0.16573365, -0.15997942],
       [ 0.6153792 ,  0.13625495, -0.79180765, -0.9024999 ]])
w3 = np.array([[-0.24242403,  0.23748292,  0.58822894, -0.22244397, -0.21516082],
       [-1.2969116 ,  0.76276046,  0.24543141, -0.17115569, -0.6496056 ],
       [-1.0891781 , -0.01295067,  0.6229789 , -0.31443328, -0.02139996],
       [-0.17632313,  0.9181154 ,  0.00322268, -0.09358749, -0.92056894]])

b1 = [0.10361356, 0.12571271, 0.08174716, -0.11954541]
b2 = [-0.10710772, 0.067965, -0.02438042, -0.07138311]
b3 = [0.007873189, -0.110927746, -0.124963015, -0.030529981, -0.03557417]

#Complete
# w1 = np.array([[ 1.5876946 , -0.6664626 ,  0.07730161, -1.1942573 ],
#        [-0.9703415 ,  1.5548829 , -2.5602245 , -1.9297166 ]])
#
# w2 = np.array([[ 3.237482  , -0.86535835, -2.176342  , -1.2469616 ],
#        [-0.24758914,  2.2084887 ,  1.7401458 ,  1.4225801 ],
#        [ 0.912785  , -1.7694603 , -2.5488098 , -1.3751976 ],
#        [-0.27932376, -0.21534976, -1.5818043 ,  0.12889428]])
#
# w3 = np.array([[-1.8401293 , -0.66504896, -0.20565084,  0.35419145, -4.166567  ],
#        [ 0.12398614, -0.48125276, -1.4355366 , -3.6758602 ,  0.20936245],
#        [-0.03913379, -2.5268989 , -1.0955528 , -2.557015  ,  1.0428941 ],
#        [ 0.19045399,  0.01802917, -3.3419023 , -4.984992  ,  0.51065886]])
#
# b1 = np.array([[ 0.5031694,   0.5708697,  -0.27838895, -0.21123832]])
# b2 = np.array([[1.2657636 , 0.6572869,  0.09734187, 0.5370622 ]])
# b3 = np.array([[-0.08192826, -0.53881526, -0.32920173, -0.6408824,  -1.2669811 ]])
numL1Val = 4
numL2Val = 5

x_train = np.array(x_train)
x_test = np.array(x_test)
def ReLu(layer, alpha):
    return np.maximum(layer, layer * alpha)

def Softmax(layer):
    return np.exp(layer) / np.sum(np.exp(layer), axis = 1, keepdims = True)

def forwardProp(alpha, input, w1, w2, w3, b1, b2, b3):
    layer1_raw = input.dot(w1) + b1
    layer1_act = ReLu(layer1_raw, alpha)
    layer2_raw = layer1_act.dot(w2) + b2
    layer2_act = ReLu(layer2_raw, alpha)
    layer3_raw = layer2_act.dot(w3) + b3
    layer3_act = Softmax(layer3_raw)
    return layer1_raw, layer1_act, layer2_raw, layer2_act, layer3_raw, layer3_act

def dReLu(layer_raw, alpha):
    deriv = np.zeros((len(layer_raw), len(layer_raw[0])))
    for i in range(len(layer_raw[0])):
        # print(deriv[i])
        if layer_raw[0][i] <= 0.0:
            deriv[0][i] = alpha
            # deriv[x][i] = 0.01
        elif layer_raw[0][i] > 0.0:
            deriv[0][i] = 1
            # deriv[x][i] = 1
        # print(deriv)
    return deriv


def backProp(alpha, w1, w2, w3, b1, b2, b3, input, startIdx, probs, layer1_act, layer1_raw, layer2_act, layer2_raw):

    probs[0][int(y_train[startIdx])] -= 1
    # probs[0][0] -= 1 #test

    dL3raw = probs
    dW3 = layer2_act.T.dot(dL3raw)
    dB3 = np.sum(dL3raw, axis = 0, keepdims = True)

###################################
    dL2raw = dL3raw.dot(w3.T) * dReLu(layer2_raw, alpha)
    dW2 = layer1_act.T.dot(dL2raw).T
    dB2 = np.sum(dL2raw, axis = 0, keepdims = True)

##################################
    dL1raw = dL2raw.dot(w2.T) * dReLu(layer1_raw, alpha)
    dW1 = dL1raw.T.dot(input).T
    dB1 = np.sum(dL1raw, axis = 0, keepdims = True)
    # learning_rate = 0.6 #best for 50 epochs and 505 samples
    # learning_rate = 0.07 #best for 50 epochs and 50 samples
    # learning_rate = 0.018 #best for 40 * 149 epochs
    learning_rate = 0.015
    # learning_rate = 0.093 #best for 30 epochs
    # print(dW3)
    # print()
    # print(dW2)
    # print()
    # print(dW1)

    w1 -= learning_rate * dW1
    w2 -= learning_rate * dW2
    w3 -= learning_rate * dW3
    b1 -= learning_rate * dB1
    b2 -= learning_rate * dB2
    b3 -= learning_rate * dB3

    return w1, w2, w3, b1, b2, b3

def loss(correctOutputs, numSamples):
    for i in range(len(correctOutputs)):
        if correctOutputs[i] == 0:
            correctOutputs[i] = 1e-7
    return np.sum(-np.log(correctOutputs)) / numSamples


# For testing the accuracy before optimization
# layer1_raw = x_test.dot(w1) + b1
# layer1_act = ReLu(layer1_raw)
# layer2_raw = layer1_act.dot(w2) + b2
# for row in layer2_raw:
#     for e in row:
#         if e > 20 or e < -20:
#             print("alert", e)
# # print(layer2_raw)
# layer2_act = Softmax(layer2_raw)
#
#
# predictions = np.argmax(layer2_act, axis = 1)
# numCorrect = 0
#
#
# for i, input in enumerate(x_test):
#     print("Input: ", input, "  Prediction: ", predictions[i], "  True/Observed: ", y_test[i])
#     if predictions[i] == y_test[i]:
#         numCorrect += 1
# print(numCorrect / len(x_test))
# model = models.load_model("PICO_Incomplete_SoftHandGesturePredicterModel2-4-4-5")

# epochs = 100
epochs = 50
# epochs = 30
alpha = 0.2
#optimal epoch = 50
batch_size = 1
for e in range(epochs):
    correctProb = []
    list = []
    accuracy = 0
    numSamples = len(x_train)
    # print(numSamples)
    # numSamples = 1
    for idx in range(numSamples):

        input = [x_train[idx][0], x_train[idx][1]]
        input = np.array([input])
        # print(input, y_train[idx])
        layer1_raw, layer1_act, layer2_raw, layer2_act, layer3_raw, layer3_act = forwardProp(alpha, input, w1, w2, w3, b1, b2, b3)
        p = np.argmax(layer3_act, axis = 1)
        if p == y_train[idx]:
            accuracy += 1
        # print(input, " Label: ", y_test[idx], " Predict: ", p)
        # print("Prediction: ", p, "Label: ", y_test[idx])
        # print(layer3_act)


        # model.predict(input)
        # print(layer1_act)
        # getLayerOutput = K.function(model.layers[0].input, model.layers[0].output)
        # # alpha constant for leaky_relu is not 0.01, it's actually 0.2
        # print(getLayerOutput([input]))
        correctProb.append(layer3_act[0][int(y_train[idx])])
        prob = layer3_act
        w1, w2, w3, b1, b2, b3 = backProp(alpha, w1, w2, w3, b1, b2, b3, input, idx, prob, layer1_act, layer1_raw, layer2_act, layer2_raw)

    # accuracy /= len(y_test)
    print("accuracy: ", accuracy/numSamples, "loss: ", loss(correctProb, numSamples))







# predictions = np.argmax(layer2_act, axis = 1)
# numCorrect = 0


# for i, input in enumerate(x_test):
#     print("Input: ", input, "  Prediction: ", predictions[i], "  True/Observed: ", y_test[i])
#     if predictions[i] == y_test[i]:
#         numCorrect += 1
# print(numCorrect / len(x_test))

