import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, MaxPooling2D
from keras.layers.convolutional import Conv2D, Cropping2D
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sample_images = []
with open('../Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        sample_images.append(line)

train_image_samples, validation_image_samples = train_test_split(sample_images, test_size=0.2)


def preProcessImage(image):
    shape = image.shape
    image = image[np.math.floor(shape[0] / 5):shape[0] - 25, 0:shape[1]]
    image = (image / 127.5) - 1
    return image


def generator(samples, batch_size=32):
    num_images = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_images, batch_size):
            batch_images = samples[offset:offset + batch_size]

            images = []
            measurements = []

            for batch_image in batch_images:
                for i in range(3):
                    measurement = float(batch_image[3])
                    correction = 0.25
                    if i == 0:
                        # Apply correction to left image
                        measurement = measurement + correction
                    if i == 2:
                        # Apply correction to right image
                        measurement = measurement - correction
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '../Data/IMG/' + filename
                    image = cv2.imread(current_path)
                    # image = preProcessImage(image)

                    images.append(image)
                    measurements.append(measurement)

                    images.append(np.fliplr(image))
                    measurements.append(-measurement)

            x_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(x_train, y_train)


train_generator = generator(train_image_samples, batch_size=32)
validation_generator = generator(validation_image_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation="relu"))
# model.add(MaxPooling2D(pool_size=(1, 1), strides=None, padding='valid', data_format=None))
# model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
print("Training images: {0}".format(len(train_image_samples)))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_image_samples) / 32,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_image_samples) / 32, epochs=3, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
# print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

exit()
