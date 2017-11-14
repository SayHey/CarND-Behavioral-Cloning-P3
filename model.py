
# Define model

from architecture import model_architecture
from data import SIZE_X, SIZE_Y

image_shape = (SIZE_X, SIZE_Y, IMG_CH)
model = model_architecture(image_shape)
model.compile(loss = 'mse', optimizer = 'adam')

# Train model

from data import data_generator
from data import load_data

train_samples, validation_samples = load_data()

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

# Save model

model.save('model.h5')