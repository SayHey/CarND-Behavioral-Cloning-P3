# Testing environment
import os
import numpy as np

# load_data test
from data import load_data

# Define history callbacks

from keras.callbacks import Callback
class LossHistory(Callback):
    def __init__(self):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


def loos_graph(history):
    plt.plot(history.losses[1:])
    plt.plot(history.val_losses[1:])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
#
import matplotlib.pyplot as plt
from data import train_generator, validation_generator

def augmented_data_distribution():
    batch_size = 1000#len(train_samples)
    batch_count = 1
    data = train_generator(train_samples, batch_size = batch_size)

    for i in range(batch_count):
        batch_images, batch_steering = (next(data))         
        plt.hist(batch_steering, bins = 100)
        plt.show()

def raw_data_distribution():
    batch_size = 1000#len(train_samples)
    batch_count = 1
    data = validation_generator(train_samples, batch_size = batch_size)

    for i in range(batch_count):
        batch_images, batch_steering = (next(data))         
        plt.hist(batch_steering, bins = 100)
        plt.show()

#train_samples, validation_samples = load_data()
#raw_data_distribution()
#augmented_data_distribution()


