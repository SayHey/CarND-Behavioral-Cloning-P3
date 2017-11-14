
### Data Augmentation

# parameters
BRIGHTNESS_RANGE = .5           # brightness range parameter used in brightness_augmentation
TRANS_X_RANGE = 100             # transition range parameter used in shift_augmentation
TRANS_Y_RANG = 40               # transition range parameter used in shift_augmentation
TRANS_ANGLE = .4                # angle offset parameter used in shift_augmentation
OFF_CENTER_ANGLE = .25          # parameter used in random_image_choice
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
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

def shift_augmentation(image, angle):  
    #
    # Randomly shifts the image along x and y axis.
    #
    x_translation = TRANS_X_RANGE * (np.random.uniform() - .5)
    y_translation = TRANS_Y_RANGE * (np.random.uniform() - .5)
    angle += x_translation * TRANS_ANGLE / TRANS_X_RANGE
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    image = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return image, angle

def random_image_choice(data_line): 
    #
    # Randomly chooses the image from the set of center, right or left images available in data.
    #
    img_choice = np.random.randint(3)
    img_path = DATA_PATH + data_line[RANDOM_IMAGE[img_choice]]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    angle = float(data_line['steering']) + OFF_CENTER_ANGLE * (img_choice - 1)
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


### Data Preprocessing

# parameters
SIZE_X = 64 # image new x dimension parameter used in crop_image
SIZE_Y = 64 # image new y dimension parameter used in crop_image

# functions
def crop_image(image):

    top_crop = 0.375*image.shape[0]
    bottom_crop = 0.875*image.shape[0]
    image = image[top_crop:bottom_crop, :, :]
    image = cv2.resize(image, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_AREA)
    return image

def preprocess_pipeline(data_line):

    image, angle = random_image_choice(data_line)
    image = crop_image(image)
    image, angle = shift_augmentation(image, angle)
    image = brightness_augmentation(image)    
    image = flip(image)    
    return image, angle

### Data Generators

# parameters
BATCH_SIZE = 32

# functions
def generate_batch(data, bias):
    
    data_length = len(data)
    batch_images = np.zeros((BATCH_SIZE, SIZE_X, SIZE_Y, 3))
    batch_steering = np.zeros(BATCH_SIZE)

    while 1:
        for batch in range(BATCH_SIZE):
            line = np.random.randint(data_length)
            data_line = data[line]
            
            image, angle = preprocess_pipeline(data_line)
            while filter_zero_angle_data(angle, bias):
               image, angle = preprocess_pipeline(data_line)

            batch_images[batch] = image
            batch_steering[batch] = angle

        yield batch_images, batch_steering