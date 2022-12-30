import variables as var
def dotProduct(arr1, arr2, product):
    """Performs a dot product between arr1 and arr2 and returns the product.
    Both arrays must be passed in as a 2D array even if they only have one
    row or column. This function performs the matrix multiplication of the
    2 arrays passed in if the 2 arrays are 2D arrays.
    If array 1 is a 2D array and array 2 is a 2D array with 1 column, the nth
    element in the product array is the sum of the products of each element
    in the nth row of the first array multiplied by the element at the
    corresponding index in the second array.
    If array 2 is a 2D array and array 1 is an array with 1 row, the nth
    element in the product array is the sum of the products of each element
    in the nth column of the second array multiplied by the element at the
    corresponding index in the first array.
    
    :param arr1: the first array, whose number of columns must match the number
        of rows in the second array
    :type arr1: list of list of float
    :param arr2: the second array, whose number of rows must match the number
        of columns in the first array
    :type arr2: list of list of float
    
    :return: the dot product of the 2 arrays, specifically the first array
        dotted by the second.
    """
    for r in range(len(arr1)):
        for c2 in range(len(arr2[0])):
            sumVal = 0
            for c in range(len(arr1[0])):
                sumVal += arr1[r][c] * arr2[c][c2]
            product[r][c2] = sumVal
            
def addBiases(layer, bias):
    """Takes a given layer and applies the biases to the raw values
    of the layer.
    
    :param layer: a hidden layer in the neural network, holding
        the raw values of that layer computed from dot product
        of the activated values of the previous layer, or the inputs
        if the previous layer was the inputs, with the current weights
    :type layer: list of list of float
    :param bias: the biases associated with the neurons of the layer
        passed in
    :type bias: list of float
    """
    
    for r in range(len(layer)):
        for c in range(len(layer[0])):
            layer[r][c] += bias[c]
    

def transpose(arr):
    '''This method transposes a given array, returning an array
    where the first row of the original array is now the first
    column of the updated array etc.
    
    :param arr: holds the array to be transposed
    :type arr: list of list of float
    
    :return: returns the transposed version of the original array
    '''
    updated = [[0 for i in range(len(arr))] for i in range(len(arr[0]))]
    # iterate through arr, updated will store element with inverse coordinates
    for r in range(len(arr)):
        for c in range(len(arr[0])):
            updated[c][r] = arr[r][c]
    
    return updated

def calcDBias(dLayer, bias):
    """Calculates the derivative of the Sparse Categorical Cross Entropy
    loss function with respect to the bias.
    
    :param dLayer: holds the derivative of the Sparse Categorical Cross Entropy
    loss function with respect to each neuron in a layer
    :type dLayer: list of list of float
    :param bias: the biases associated with the layer
    :type bias: list of float
    """
    
    for b in range(len(bias)):
        bias[b] -= var.learning_rate * dLayer[0][b]

def applyDWeights(dWeights, weights):
    """Applies the gradients or the derivative of the Sparse Categorical
    Cross Entropy loss function with respect to each weight of a layer
    multiplied by the learning rate, to the corresponding weight in that layer.
    
    :param dWeights: the derivative of the Sparse Categorical Cross Entropy loss
        function with respect to each weight for a specific layer
    :type dWeights: list of list of float
    :param weights: the weights of a layer
    :type weights: list of list of float
    """
    for r in range(len(dWeights)):
        for c in range(len(dWeights[0])):
            weights[r][c] -= (dWeights[r][c] * var.learning_rate)
    
def printMatrix(matrix):
    """Used for debugging. Prints out a 2D array by printing a row per line.
    
    :param matrix: the 2D array to be printed
    :type matrix: list of list of float
    """
    for r in matrix:
        print(r)
        