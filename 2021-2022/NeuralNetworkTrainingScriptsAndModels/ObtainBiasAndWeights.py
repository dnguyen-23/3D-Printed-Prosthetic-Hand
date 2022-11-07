from preparation import *
import math
model = models.load_model("PICO_Incomplete_SoftHandGesturePredicterModel2-4-4-5")


def printStructure():
    for layer in model.layers:
        print(layer.name, layer)

def predictFromModel():
    predictions = model.predict(x_test)
    numcorrect = 0
    for i in range(len(predictions)):
        print("Prediction: ", np.argmax(predictions[i]), "Actual: ", y_test[i])
        if np.argmax(predictions[i]) == y_test[i]:
            numcorrect += 1
    print(numcorrect / len(predictions))

printStructure()
allWeights = []
allBiases = []
for l in model.layers:
    allWeights.append((l.get_weights()[0]))
    # get_weights() returns the weights as and the second thing that is return are the biases
    # so you only want get_weights()[0]
    allBiases.append(l.bias.numpy())

print(allWeights)
print(len(allWeights))

# Printing the weights DO NOT DELETE
# for arr in allWeights:
#     for row in arr:
#         for elem in row:
#             print(elem, ", ")
#     print("\n")

for biases in allBiases:
    for b in biases:
        print(b, end = ", ")

    print()
predictFromModel()
# input = np.array([1, 2])
# layer1 = np.dot(input, allWeights[0]) + allBiases[0]
# print(layer1)
# layer2 = np.dot(layer1, allWeights[1]) + allBiases[1]
# print(layer2)
# layer3 = np.dot(layer2, allWeights[2]) + allBiases[2]
# print(layer3)
# layer4 = np.dot(layer3, allWeights[3]) + allBiases[3]
# print(layer4)
# layer5 = np.dot(layer4, allWeights[4]) + allBiases[4]
# print(layer5)
# print(math.exp(709))
# numInputs = 45
# for i in range(len(x_test)):
#     print(x_test[i][0], end = ", ")
#     print(x_test[i][1], end = ", ")
#     if i == numInputs:
#         break
#
# print(len(x_test))
# print("\n")
# for x in range(len(y_test)):
#     print(y_test[x], end = ", ")
#     if x == numInputs:
#         break

# myListX = []
# myListY = []
# sets = []
# for i in range(5):
#     dataFnd = 0
#     for x in range(len(y_test)):
#         if y_test[x] == i:
#             dataFnd += 1
#             curSet = [x_test[x][0], x_test[x][1], y_test[x]]
#             sets.append(curSet)
#             # myListX.append(x_test[x])
#             # myListY.append(y_test[x])
#         if dataFnd == 9:
#             break
#
# random.shuffle(sets)
#
# for s in sets:
#     print("[", s[0] / 1000, ", ", s[1] / 1000, "]", end = ", \n")
#
# for s in sets:
#     print(s[2], end = ", ")
# for s in sets:
#     myListX.append(s[0])
#     myListX.append(s[1])
#     myListY.append(s[2])
#
# for x in myListX:
#     print(x, end = ", ")
#
# for y in myListY:
#     print(y, end = ", ")
