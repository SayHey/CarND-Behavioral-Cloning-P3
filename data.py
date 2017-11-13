# Data augmentation

BRIGHTNESS_RANGE = .5
TRANS_X_RANGE = 100
TRANS_Y_RANG = 40
TRANS_ANGLE = .2
OFF_CENTER_IMG = .25
DATA_PATH = 'data/'

def flip_augmentation(image, angle):

    if np.random.randint(2) == 0:
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def brightness_augmentation(image):

    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

def shift_augmentation(image):

    x_translation = TRANS_X_RANGE * np.random.uniform() - TRANS_X_RANGE / 2
    y_translation = TRANS_Y_RANGE * np.random.uniform() - TRANS_Y_RANGE / 2
    new_angle = angle + 2 * x_translation * TRANS_ANGLE / TRANS_X_RANGE
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

def random_image_choice(line):

    random = { 0 : 'left',
               1 : 'center',
               2 : 'right'}
    img_choice = np.random.randint(3)
    img_path = DATA_PATH + random[img_choice]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    angle = float(line['steering']) + OFF_CENTER_IMG * (img_choice - 1)
    return image, angle

def filter_zero_angle_data(angle, bias):
    
    threshold = np.random.uniform()
    if (abs(angle) + bias) < threshold:
        return None, None

# ---------------

# Data Preprocessing

SIZE_X = 64
SIZE_Y = 64

def cropImage(image):

    top_crop = 0.375*image.shape[0]
    bottom_crop = 0.875*image.shape[0]
    image = image[top_crop:bottom_crop, :, :]
    image = cv2.resize(image, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_AREA)    
    return np.resize(resize, (1, SIZE_X, SIZE_Y, IMG_CH))

# ---------------

# Data Generators
