# Define model

from architecture import model_architecture
from data import SIZE_X, SIZE_Y, IMG_CH

image_shape = (SIZE_Y, SIZE_X, IMG_CH)
model = model_architecture(image_shape)

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])

# Train model

from data import load_data, plain_data

# Load data
train_samples, validation_samples = load_data()
train_images, train_angles = plain_data(train_samples)
validation_images, validation_angles = plain_data(validation_samples)

# Define parameters
batch_size = 64
nb_epoch = 5

history = model.fit(train_images, train_angles,
                    batch_size = batch_size, 
                    nb_epoch = nb_epoch,
                    verbose = 1, 
                    validation_data = (validation_images, validation_angles))

# Save model

model.save('model.h5')
