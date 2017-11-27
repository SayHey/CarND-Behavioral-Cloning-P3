# Testing environment
import os
import numpy as np
from matplotlib import pyplot as plt

# load_data test
from data import load_data
train_samples, validation_samples = load_data()

# pipeline test
from data import random_image_choice, shift_augmentation, brightness_augmentation, flip_augmentation, preprocess_pipeline
def pipeline_test():
    data_length = len(train_samples)

    for i in range(10):
        line = np.random.randint(data_length)
        data_line = train_samples[line]

        #image, angle = random_image_choice(data_line)
        #image, angle = shift_augmentation(image, angle)
        #image = brightness_augmentation(image)    
        #image, angle = flip_augmentation(image, angle)  
    
        image, angle = preprocess_pipeline(data_line)

        print(angle)
        plt.imshow(image, interpolation='nearest')
        plt.show()    
        os.system('cls')

pipeline_test()


# data_generator test
from data import train_generator
def data_generator_test():
    batch_size = 1024
    batch_count = 2
    data = train_generator(train_samples, batch_size = batch_size)

    for i in range(batch_count):
        batch_images, batch_steering = (next(data))
        for (image, angle) in zip(batch_images, batch_steering):
            print(angle)
            plt.imshow(image, interpolation='nearest')
            plt.show()    
            os.system('cls')

data_generator_test()

# plain_data test
def plain_data_test():
    images, angles = plain_data(train_samples)
    print(angles[0])
    plt.imshow(images[0], interpolation='nearest')
    plt.show()    
    os.system('cls')


from keras.models import load_model
from data import validation_generator

def model_test():

    size = 512

    model = load_model('model.h5')
    train_samples, validation_samples = load_data()
    a = len(train_samples)
    b = len(validation_samples)
    data = validation_generator(train_samples, batch_size = size)
    augmented_data = train_generator(train_samples, batch_size = size)

    images, steering = (next(data))
    aug_images, aug_steering = (next(augmented_data))
    predicted_steering = model.predict_generator(data, val_samples = 1)

    bins = np.linspace(-0.5, 0.5, 100)

    plt.hist(predicted_steering, 100)
    plt.show()

    plt.hist(steering, bins, alpha=0.5, label='steering')
    plt.hist(aug_steering, bins, alpha=0.5, label='augmented steering')
    plt.hist(predicted_steering, bins, alpha=0.5, label='predicted steering')
    plt.legend(loc='upper right')
    plt.show()


model_test()