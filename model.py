import csv
import cv2
import numpy as np

lines = []
with open('../Data/archivedData/dataSet2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        measurement = float(line[3])
        correction = 0.2
        if i == 1:
            # Apply correction to left image
            measurement = measurement + correction
        if i == 2:
            # Apply correction to right image
            measurement = measurement - correction
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../Data/archivedData/dataSet2/IMG/' + filename
        image = cv2.imread(current_path)

        images.append(image)
        measurements.append(measurement)

        images.append(np.fliplr(image))
        measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((32, 20), (0, 0))))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
exit()
