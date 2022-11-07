import variables as var
import matrixOps as mOps
import forwardPass as fP
def dRelu(layer_raw, alpha, deriv):
    for i in range(len(layer_raw)):
        for j in range(len(layer_raw[0])):
            if layer_raw[i][j] <= 0.0:
                deriv[i][j] *= alpha
            elif layer_raw[i][j] > 0.0:
                deriv[i][j] *= 1
    
def backProp(inputs, labelIdx):
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
    dRelu(var.layer2_raw, var.reluCoef, dL2raw)
    
    mOps.dotProduct(mOps.transpose(var.layer1_act), dL2raw, dW2)
    mOps.calcDBias(dL2raw, var.b2)
    
    
    # Getting derivative of loss with respect to derivative of the raw first layer
    mOps.dotProduct(dL2raw, mOps.transpose(var.w2), dL1raw)
    dRelu(var.layer1_raw, var.reluCoef, dL1raw)
    
    mOps.dotProduct(mOps.transpose(inputs), dL1raw, dW1)
    mOps.calcDBias(dL1raw, var.b1)
    
    mOps.applyDWeights(dW1, var.w1)
    mOps.applyDWeights(dW2, var.w2)
    mOps.applyDWeights(dW3, var.w3)
# 
# # void testAccuracy()
# # {
# #   float testingInputs[] = {0.0, 0.461, 0.162, 0.0, 0.202, 0.0, 0.115, 0.118, 0.135, 0.552, 0.12, 0.132, 0.019, 0.0, 0.307, 0.0, 0.274, 0.267, 0.395, 0.0};
# #   int testingLabels[] = {4, 2, 3, 1, 4, 1, 2, 3, 1, 0};
# #   int numTestingInputs = sizeof(testingLabels) / sizeof(testingLabels[0]);
# #   int idxMidInputIdx = 0;
# #   int rPInputIdx = 1;
# #   int numCorrect = 0;
# #   for (int i = 0; i < numTestingInputs; i++)
# #   {
# #     float input[]= {testingInputs[idxMidInputIdx], testingInputs[rPInputIdx]};
# #     int gestureIdx = makePrediction(input)
# #     if (gestureIdx == testingLabels[i])
# #     {
# #       numCorrect++;
# #     }
# # 
# #     if ((idxMidInputIdx) < (numTestingInputs * 2))
# #     {
# #       idxMidInputIdx += 2;
# #     }
# # 
# #     if ((rPInputIdx + 2) < (numTestingInputs * 2))
# #     {
# #       rPInputIdx += 2;
# #     }
# # 
# # 
# #     delay(250);
# #   }
# # 
# #   float accuracy = ((float) numCorrect) / numTestingInputs;
# #   Serial.println(accuracy);
# # 
# # }