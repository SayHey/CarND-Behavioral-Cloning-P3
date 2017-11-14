# Testing environment

from data import data_generator
from data import load_data

train_samples, validation_samples = load_data()

data = data_generator(train_samples, non_zero_bias = 1, batch_size = 3)

for i in range(10):
    data_sample = (next(data))