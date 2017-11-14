# Testing environment
import os
import numpy as np
from matplotlib import pyplot

# load_data test
from data import load_data
train_samples, validation_samples = load_data()

# pipeline test
from data import random_image_choice, shift_augmentation, brightness_augmentation, flip_augmentation, preprocess_pipeline
def pipeline_test():
    data_length = len(train_samples)

    for i in range(0):
        line = np.random.randint(data_length)
        data_line = train_samples[line]

        #image, angle = random_image_choice(data_line)
        #image, angle = shift_augmentation(image, angle)
        #image = brightness_augmentation(image)    
        #image, angle = flip_augmentation(image, angle)  
    
        image, angle = preprocess_pipeline(data_line)

        print(angle)
        pyplot.imshow(image, interpolation='nearest')
        pyplot.show()    
        os.system('cls')


# data_generator test
from data import data_generator, plain_data
def data_generator_test():
    batch_size = 0
    batch_count = 0
    data = data_generator(train_samples, non_zero_bias = 1, batch_size = batch_size)

    for i in range(batch_count):
        batch_images, batch_steering = (next(data))
        for (image, angle) in zip(batch_images, batch_steering):
            print(angle)
            pyplot.imshow(image, interpolation='nearest')
            pyplot.show()    
            os.system('cls')

# plain_data test
def plain_data_test():
    images, angles = plain_data(train_samples)
    print(angles[0])
    pyplot.imshow(images[0], interpolation='nearest')
    pyplot.show()    
    os.system('cls')