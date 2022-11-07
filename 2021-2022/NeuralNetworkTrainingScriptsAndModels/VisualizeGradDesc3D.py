file = open("/Users/danie/OneDrive/Documents/HandGestureCSVFiles_Ver2/150FlexFist.csv")
title = file.readline()
recording = file.readline()
print(len(recording))
data = []
gestures = {0 : "flex fist",
            1 : "flex index",
            2 : "flex index and middle",
            3 : "flex middle",
            4 : "flex ring and pinky"}
print(title)
print(recording)
firstComma = recording.find(",")
secondComma = recording.find(",", firstComma + 1)
data.append(float(recording[0: recording.find(",")]))
data.append(float(recording[recording.find(",") + 1: recording.find(",", recording.find(",") + 1)]))
print(gestures.items())
for key, val in gestures.items():
    print(len(val), len(recording[secondComma + 1: len(recording) - 1]))
    # print(val, recording[secondComma + 1: len(recording) - 1])
    if val == recording[secondComma + 1: len(recording) - 1]:
        print(key)
print(data)

a = [[1, 2, 3],
     [1, 2, 3]]
print(type(a))
# print(recording[recording.find(",", recording.find(",")), len(recording)])