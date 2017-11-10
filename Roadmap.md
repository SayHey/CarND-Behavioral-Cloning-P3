# **Project Roadmap** 

### The roadmap based on project material and slack pinned posts
* Mohan Karthik's Slack post [Cloning a car to mimic human driving](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff)
* Max Ritter's Slack post [End-to-end driving](https://github.com/maxritter/SDC-End-to-end-driving)
* Kiki Jewell's Slack post containing othe ruseful links [SDC Proj3 — for impatient people](https://medium.com/@kikiorgg/sdc-proj3-for-impatient-people-aad78bb0dc99)
* Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)
---

## 1. Collect data

* **Use an analog input (joystick, wheel)**
* **Need about 40k samples**

## 2. Analize and visualize data

* **Plot Angles over time**
* **Plot Angle distribution**
* **Plot Sample images**
* **Visualize data after the augmentation (angle distribution and samples)**

## 3. Augmentat data

* **subtract a static offset from the angle when choosing the left / right image**
    ```python
    img_choice = np.random.randint(3)
    if img_choice == 0:
        img_path = os.path.join(PATH, df.left.iloc[idx].strip())
        angle += OFF_CENTER_IMG
    elif img_choice == 1:
        img_path = os.path.join(PATH, df.center.iloc[idx].strip())
    else:
        img_path = os.path.join(PATH, df.right.iloc[idx].strip())
        angle -= OFF_CENTER_IMG
    ```

* **flipping the image**
    ```python
    if np.random.randint(2) == 0:
        img = np.fliplr(img)
        new_angle = -new_angle
    ```

* **Changing brightness**
    ```python
    def augment_brightness_camera_images(image):
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Compute a random brightness value and apply to the image
        # BRIGHTNESS_RANGE = .5
        brightness = BRIGHTNESS_RANGE + np.random.uniform()
        temp[:, :, 2] = temp[:, :, 2] * brightness
        # Convert back to RGB and return
        return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
    ```

* **Horizontal and vertical shifts**
    ```python
    # Compute X translation
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE
    # Randomly compute a Y translation
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)
    # Form the translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    # Translate the image
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    ```

* **Biasing towards non-0 value**
    ```python
    # Choose left / right / center image and compute new angle
    # Do translation and modify the angle again
    # Define a random threshold for each image taken
    threshold = np.random.uniform()
    # If the newly augmented angle + the bias falls below the threshold
    # then discard this angle / img combination and look again
    if (abs(angle) + bias) < threshold:
    return None, None
    ```

* **Shadow augmentation**

## 4. Preprocess data

* **remove the top 60 pixels (past the horizon) and the bottom 20 pixels (the hood of the car)**
    ```python
    roi = img[60:140, :, :]
    ```

* **Resize the image**
    ```python
    resize = cv2.resize(roi, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)
    return np.resize(resize, (1, IMG_ROWS, IMG_COLS, IMG_CH))
    ```

## 5. Design architecture

* **Nvidia pipeline or**
* **VGG16 pre-trained model**
* **lambda layers on the top to normalize the data on the fly**
    ```python
    model.add(Lambda(lambda x: x/127.5 - .5,
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CH),
                 output_shape=(IMG_ROWS, IMG_COLS, IMG_CH)))
    ```
* **color space conversion layer** 
    ```python
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    ```
* **Dropouts in all the fully connected layers.**

## 6. Train model

* **Use generators!**
* **Optimizer: Adam with a learning rate of 1e-5**
* **Slowly reduce angle bias**
    ```python
    bias = 1. / (num_runs + 1.)
    ```

## 7. Enhance model

* **Possibly multiply the predicted angle by a constan to allow sharp turns**






