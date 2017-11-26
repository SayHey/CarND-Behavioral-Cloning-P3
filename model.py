
# Define model

from architecture import model_architecture
from data import SIZE_X, SIZE_Y, IMG_CH

image_shape = (SIZE_Y, SIZE_X, IMG_CH)
model = model_architecture(image_shape)

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])

# Train model

from data import train_generator, validation_generator
from data import load_data

# Load data
train_samples, validation_samples = load_data()

# Define parameters
batch_size = 64
nb_epoch = 40
samples_per_epoch = len(train_samples)
nb_val_samples = len(validation_samples)

steps_per_epoch = 100 #int( len(train_samples) / batch_size )
validation_steps = 10 #int( len(validation_samples) / batch_size )

# Define history callbacks
from visualisation import LossHistory
history = LossHistory()

# Fit
print("Training network..")
for epoch in range(nb_epoch):

    # Fit one epoch 
    non_zero_bias = 1/(1 + epoch / 5.)
    #non_zero_bias = 1.
    print("Non zero bias = " + str(non_zero_bias))    
    
    # Define data generators 
    train = train_generator(train_samples, batch_size, non_zero_bias)
    validation = train_generator(validation_samples, batch_size)

    model.fit_generator(train,
                        steps_per_epoch = steps_per_epoch, 
                        initial_epoch = epoch,
                        epochs = epoch + 1, 
                        verbose = 1,
                        validation_data = validation, 
                        validation_steps = validation_steps,                         
                        callbacks=[history])

print("Network trained!")

# Plot loss graph
from visualisation import loos_graph
loos_graph(history)

# Save model

model.save('model.h5')
