
# Load data

import csv
import cv2
import numpy as np

images = []
measurements = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        image_path = 'data/' + line['center']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line['steering'])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
image_shape =  X_train.shape[1:4]

# Define model

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = image_shape))
model.add(Dense(1))

# Train model

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 4)

model.save('model.h5')