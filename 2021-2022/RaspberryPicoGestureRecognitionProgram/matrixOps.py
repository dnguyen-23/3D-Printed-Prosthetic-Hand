import variables as var
def dotProduct(arr1, arr2, result):
    for r in range(len(arr1)):
        for c2 in range(len(arr2[0])):
            sumVal = 0
            for c in range(len(arr1[0])):
                sumVal += arr1[r][c] * arr2[c][c2]
            result[r][c2] = sumVal
            
def addBiases(layer, bias):
    for r in range(len(layer)):
        for c in range(len(layer[0])):
            layer[r][c] += bias[c]
            
'''This method transposes a given array
    Rows of the original array become the columns of the updated array
    Updated array must have dimensions transposed
    Preconditions: takes a 2d list/array
    Postconditions: rows of the original become the columns of the updated'''

def transpose(arr):
    updated = [[0 for i in range(len(arr))] for i in range(len(arr[0]))]
    # iterate through arr, updated will store element with inverse coordinates
    for r in range(len(arr)):
        for c in range(len(arr[0])):
            updated[c][r] = arr[r][c]
    
    return updated

def calcDBias(dLayer, bias):
    for b in range(len(bias)):
        bias[b] -= var.learning_rate * dLayer[0][b]

def applyDWeights(dWeights, weights):
    for r in range(len(dWeights)):
        for c in range(len(dWeights[0])):
            weights[r][c] -= (dWeights[r][c] * var.learning_rate)
    
def printMatrix(matrix):
    for r in matrix:
        print(r)
        