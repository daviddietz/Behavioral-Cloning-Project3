import csv
import cv2
import numpy as np

lines = []
with open('../Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for lines in lines:
    source_path = lines[0]
    filename = source_path.split('/')[-1]
    current_path = '../Data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    images.append(np.fliplr(image))

    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(6, (5, 5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

model.save('model.h5')
exit()