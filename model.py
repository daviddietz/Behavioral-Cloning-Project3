import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sample_images = []
with open('../Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        sample_images.append(line)

train_image_samples, validation_image_samples = train_test_split(sample_images, test_size=0.2)
BATCH_SIZE = 128


def generator(samples, batch_size=BATCH_SIZE):
    num_images = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_images, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                for imageCameraAngle in range(3):
                    measurement = float(batch_sample[3])
                    correction = 0.25
                    if imageCameraAngle == 1:
                        # Apply correction to left image
                        measurement = measurement + correction
                    if imageCameraAngle == 2:
                        # Apply correction to right image
                        measurement = measurement - correction
                    source_path = batch_sample[imageCameraAngle]
                    filename = source_path.split('/')[-1]
                    current_path = '../Data/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    images.append(image)
                    measurements.append(measurement)

                    images.append(np.fliplr(image))
                    measurements.append(-measurement)
            x_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(x_train, y_train)


train_generator = generator(train_image_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_image_samples, batch_size=BATCH_SIZE)

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (3, 3), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=100))
model.add(Dense(units=50))
model.add(Dropout(0.5))
model.add(Dense(units=10))
model.add(Dense(units=1))
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_image_samples),
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_image_samples) / BATCH_SIZE, epochs=5, verbose=1)

model.save('model.h5')

## print the keys contained in the history object
# print(history_object.history.keys())

## plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

exit()
