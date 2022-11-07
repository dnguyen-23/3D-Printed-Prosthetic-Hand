from preparation import *
# This is a sample Python script.

# x_train = list(zip(a, b))
model = KNeighborsClassifier(n_neighbors = 4)

# find the unique values for y_train and then turn them into numerical values



# creating the dictionary/hashmap; this can be done because the unique values and the numerical values are in
# corresponding order

# print(map)
# print(y_train_vocab, numerical_y_train)

# for i in range(len(y_train)):
#     y_train[i] = map[y_train[i]]
# print(y_train)
# print(categorical_y_train)
# model.fit(x_train, y_train)


# testing_data = pd.read_csv("/Users/danie/OneDrive/Documents/HandGestureCSVFiles/TestingGestures.csv")
# x_testx = testing_data['Index Finger']
# x_testy = testing_data['Ring Pinky Finger']
# x_test = list(zip(x_testx, x_testy))
# predictions = model.predict(x_test)
#
#
# for i in predictions:
#     print(i)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#