
import cv2
import numpy as np

### Data Augmentation

# parameters
SIZE_X = 320                     # image x dimension parameter
SIZE_Y = 160                    # image y dimension parameter
IMG_CH = 3                      # image number of chanels
BRIGHTNESS_RANGE = .5           # brightness range parameter used in brightness_augmentation
TRANS_X_RANGE = 100             # transition range parameter used in shift_augmentation
TRANS_Y_RANGE = 40              # transition range parameter used in shift_augmentation
TRANS_ANGLE = .4                # angle offset parameter used in shift_augmentation
OFF_CENTER_ANGLE = .1           # parameter used in random_image_choice
DATA_PATH = 'data/'             # data path used in random_image_choice
RANDOM_IMAGE = { 0 : 'left',    # switch case dictionary for random_image_choice
                 1 : 'center',
                 2 : 'right'} 

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

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

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
    img_path = DATA_PATH + data_line[RANDOM_IMAGE[img_choice]]

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    angle = float(data_line['steering']) + OFF_CENTER_ANGLE * (1 - img_choice)
    return image, angle

def filter_zero_angle_data(angle, bias): 
    #
    # Randomly drops images with low steering angle values
    # based on bias value that is adjusted during training.
    # Bias value decreases from 1.0 to 0.0 increasing 
    # the probability to drop all the lower angles
    #
    threshold = np.random.uniform()
    return (abs(angle) + bias) < threshold

def preprocess_pipeline(data_line):

    image, angle = random_image_choice(data_line)
    image, angle = shift_augmentation(image, angle)
    image = brightness_augmentation(image)    
    image, angle = flip_augmentation(image, angle)    
    return image, angle


### Data Generator
import os
from matplotlib import pyplot
def data_generator(data, non_zero_bias = 1.0, batch_size = 32):
    
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


### Load Data

from csv import DictReader

def load_data():
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = DictReader(csvfile, skipinitialspace = True)
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    return train_test_split(samples, test_size=0.1)