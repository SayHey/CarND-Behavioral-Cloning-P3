
# Load data

from csv import DictReader

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = DictReader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

# Define model

from architecture import model_architecture
from data import SIZE_X, SIZE_Y

image_shape = (SIZE_X, SIZE_Y)
model = model_architecture(image_shape)
model.compile(loss = 'mse', optimizer = 'adam')

# Train model

from data import data_generator

batch_size = 32
nb_epoch = 5
samples_per_epoch = len(train_samples)
nb_val_samples = len(validation_samples)

for epoch in range(nb_epoch):

    # Define data generators
    non_zero_bias = 1. / (epoch + 1.)
    train_generator = data_generator(train_samples, non_zero_bias = non_zero_bias, batch_size = batch_size)
    validation_generator = data_generator(validation_samples)

    # Fit one epoch
    model.fit_generator(train_generator, 
                        samples_per_epoch = samples_per_epoch, 
                        validation_data = validation_generator, 
                        nb_val_samples = nb_val_samples, 
                        nb_epoch = 1)
    

model.save('model.h5')