
import cv2
import numpy as np

# parameters
SIZE_X = 320                     # image x dimension parameter
SIZE_Y = 160                    # image y dimension parameter
IMG_CH = 3                      # image number of chanels
BRIGHTNESS_RANGE = .5           # brightness range parameter used in brightness_augmentation
TRANS_X_RANGE = 100             # transition range parameter used in shift_augmentation
TRANS_Y_RANGE = 40              # transition range parameter used in shift_augmentation
TRANS_ANGLE = .4                # angle offset parameter used in shift_augmentation
OFF_CENTER_ANGLE = .25           # parameter used in random_image_choice
DATA_PATHS = {'data1/',           # data path used in random_image_choice
              'data2/',
              'data3/',
              'data4/',
              'data5/',
              'data6/'}
IMAGE_MAP = { 'center' : 0,     # switch case dictionary for random_image_choice
              'left' : 1,
              'right' : 2} 


### Load Data

import csv
from sklearn.model_selection import train_test_split

def load_data():
    samples = []
    for data in DATA_PATHS:
        with open(data + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile, skipinitialspace = True)
            for line in reader:
                samples.append(line)
    return train_test_split(samples, test_size=0.1)

def read_line(data_line, img_choice):

    img_path = data_line[img_choice]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_choice = (img_choice + 1) % 3
    angle = float(data_line[3]) + OFF_CENTER_ANGLE * (img_choice - 1)
    return image, angle
    

### Data Augmentation


# functions
def flip_augmentation(image, angle):
    #
    # Randomly flips the image.
    #
    if np.random.randint(2) == 0:
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def brightness_augmentation(image):    
    #
    # Randomly changes the brightness.
    #
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * brightness
    image[:, :, 2][image[:, :, 2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def shift_augmentation(image, angle):  
    #
    # Randomly shifts the image along x and y axis.
    #
    x_translation = TRANS_X_RANGE * (np.random.uniform() - .5)
    y_translation = TRANS_Y_RANGE * (np.random.uniform() - .5)
    angle += x_translation * TRANS_ANGLE / TRANS_X_RANGE
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return image, angle

def random_image_choice(data_line): 
    #
    # Randomly chooses the image from the set of center, right or left images available in data.
    #
    img_choice = np.random.randint(3)
    return read_line(data_line, img_choice)

def filter_zero_angle_data(angle, bias): 
    #
    # Randomly drops images with low steering angle values
    # based on bias value that is adjusted during training.
    # Bias value decreases from 1.0 to 0.0 increasing 
    # the probability to drop all the lower angles
    #
    threshold = (np.random.uniform() - bias) * 0.1
    return abs(angle) < threshold 

def preprocess_pipeline(data_line):

    #image, angle = random_image_choice(data_line)
    image, angle = read_line(data_line, IMAGE_MAP['center'])
    image, angle = shift_augmentation(image, angle)
    image = brightness_augmentation(image)    
    image, angle = flip_augmentation(image, angle)    
    return image, angle


### Data Generators

def train_generator(data, batch_size = 32, non_zero_bias = 1.0):
    #
    # Generates augmented training data
    #      
    data_length = len(data)
    while 1:    
        images = [] 
        angles = []
        for batch in range(batch_size):
            line = np.random.randint(data_length)
            data_line = data[line]            
            image, angle = preprocess_pipeline(data_line)
            while filter_zero_angle_data(angle, non_zero_bias):
               image, angle = preprocess_pipeline(data_line)
            images.append(image)
            angles.append(angle)        
        yield np.array(images), np.array(angles)
       

from matplotlib import pyplot as plt
def validation_generator(data, batch_size = 32):
    #
    # Generates validation data
    #      
    data_length = len(data)    
    while 1:
        images = [] 
        angles = []
        for batch in range(batch_size):
            line = np.random.randint(data_length)
            image, angle = read_line(data[line], IMAGE_MAP['center'])
            images.append(image)
            angles.append(angle)

        yield np.array(images), np.array(angles)

def plain_data(data):
    #
    # Returns the entire data set of 'center' images
    #
    images = []
    angles = []
    for data_line in data:
        image, angle = read_line(data_line, IMAGE_MAP['center'])
        images.append(image)
        angles.append(angle)

    return np.array(images), np.array(angles)