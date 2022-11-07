from preparation import *
import netron

# classifier = models.load_model("Pico_Complete_SoftHandGesturePredicterModel2-4-4-5")
# predictions = classifier.predict(x_test)
# netron.start("C:/Users/danie/PycharmProjects/GestureFingerAcutationMLmodel/PICO_SoftHandGesturePredicterModel2-4-4-5.tflite")
# numepochs = 5
# for x in range(4):
#     idx = x + 1

numepochs = 40
losses = np.zeros((1, numepochs))
acc = np.zeros((1, numepochs))
losses = np.array(losses)
acc =  np.array(acc)
for i in range(1): #5 models
    classifier = Sequential()
    classifier.add(Dense(units = 3, activation = "leaky_relu", input_dim = 3))
    # classifier.add(Dense(units = 5, activation = "relu"))
    classifier.add(Dense(units = 3, activation = "leaky_relu"))
    classifier.add(Dense(units = 3, activation = "leaky_relu"))
    # classifier.add(Dense(units = 3, activation = "leaky_relu"))
    classifier.add(Dense(units = 5, activation = "softmax"))
    classifier.compile(optimizer = "rmsprop", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), metrics = ['accuracy'])
    history = classifier.fit(x_train, y_train, batch_size = 1, epochs = numepochs) #history stores the information about the training process while the classifier trains itself
    predictions = classifier.predict(x_test)
    losses = np.vstack([losses, history.history['loss']])
    acc = np.vstack([acc, history.history['accuracy']])
# converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
# tfliteModel = converter.convert()
# with open("PICO_SoftHandGesturePredicterModel2-4-4-5.tflite", "wb") as file:
#     file.write(tfliteModel)
# # #     print(tf.keras.backend.eval(classifier.optimizer.lr))
# print(classifier.summary())


numCorrect = 0
for i, elem in enumerate(predictions):
    print("Predictions:", np.argmax(elem), " Actual:", y_test[i])
    if np.argmax(elem) == y_test[i]:
        numCorrect += 1
print(numCorrect / len(predictions))

if (numCorrect / len(predictions) > 0.80):
    print("saved")
    classifier.save("PICO_Incomplete_HandGestureSyntheticInputs3-3-3-5")
#
# # a = numCorrect / len(predictions)
# losses = np.delete(losses, 0, 0)
# acc = np.delete(acc, 0, 0)
# # axis = 0 in order to average up the rows
# losses = np.mean(losses, axis = 0)
# acc = np.mean(acc, axis = 0)
# for a in acc:
#     print(a, end = " ")
#
# print("\n")
#
# for l in losses:
#     print(l, end = " ")
#     # losses = np.mean(losses, axis = 1)
