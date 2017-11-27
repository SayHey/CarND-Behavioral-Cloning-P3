# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualisations/network.png "Model Visualization"

[image2]: ./visualisations/raw_data.png "Raw Data"
[image3]: ./visualisations/aug_data_1bias.png "Aug Data 1"
[image4]: ./visualisations/aug_data_0bias.png "Aug Data 0"
[image5]: ./visualisations/predicted.png "Predicted Data 0"
[image6]: ./visualisations/combined.png "Combined Data"


[image7]: ./visualisations/figure_1.png "Aug Image 1"
[image8]: ./visualisations/figure_1-1.png "Aug Image 2"
[image9]: ./visualisations/figure_1-2.png "Aug Image 3"
[image10]: ./visualisations/figure_1-3.png "Aug Image 4"

[image11]: ./visualisations/figure_1-3.png "Aug Image 4"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

I organized the code in the following way:

* The data.py file contains the defenitions of all the data related functions such as extracting data from csv, data augmentation functions, data augmentation pipeline and data generators for training and validating the model.
* The architecture.py contins the Keras defenition of convolution neural network architecture used inside the model
* The model.py file contains the code for training and saving the convolution neural network.

All files contain comments to explain how the code works. There are also some python files that are not related to the submission such as test.py (used for testing) and visualisation.py (used for generating various graphs).


### Model Architecture and Training Strategy

#### 1. Gathering training data

I didn't succseed in training the network on data from keyboard input, so I end up using gaming steering wheel. The data turn out to be much more smooth and informative. 
Eventually I ran tens of laps and collected about 30000 samples. I used data only from the first track, but I drove in both directions. I also applied different styles of driving: 
* drive perfectly in the center of the road as much as I can
* recovery zig-zag drive
* "race-like" drive smoothing corners

#### 2. Analyzing training data

Let's analyze the distribution of steering angles in raw data:

![alt text][image2]

As we can see there are two issues with this data:

* There are a lot of small and zero steering angle measurments in the data, which is bad, because the model trained on such data tend to be biased towards predicting zeros. This would result in difficulties in driving around sharp corners.
* Data is slightly asymmetrical, because even though there are samples from both clockwise and counterclockwise runs, there are more data from counterclockwise laps. This sis bad, because the model tend to be biased towards predicting slightly negative angles even on straight segments.

Both these issues were adressed during data processing step.

#### 3. Processing training data

I used Keras generators for data preparation as it helped to optimize training process.
My initial plan for data processing pipeline was the folowing:

* randomly choose sample from the data set
* randomly choose center, left, or right image from the sample and adjust corresponding steering angle by substracting constant valuee for right image or adding for the left image
* perform shift augmentation by randomly shifting the image along horizontal and vertical axis and adjust steering angle according to the shift
* perform brightness augmentation by randomly changing brightness
* randomly flip the image along horizontal axis and multiply steering angle by -1 to adress data asymmetry issue mentioned above
* randomly filter zero data (more about that below)

But I faced some difficulties when using left and right images. For some reason my model didn't train well, the loss fluctuated and the final model performed bad. I waste a lot of time trying to figure out why, but didn't succseed. So I ended up abandoning using randomized image choosing, colected more data and used only center images.

Here are some examples of post augmentation images:

![alt text][image8]
![alt text][image10]
![alt text][image9]

To eliminate the issue of zero bias in data I implemented a system that randomly filters all the samples that have a small steering angle according to some threshold. This threashold is parametrised with some bias value that changes during training.
```sh
non_zero_bias = 1 / (1 + .2 * epoch)
```
The bias parameter equals 1 during first epoch, thus all samples with every steering angle pass the filter. From epoch to epoch it gradualy decreases filtering more and more samples with small steering angles.

#### 2. Analyzing training data

Here is the final data distribution:

* For zero bias filter parameter equals 1 (first epoch of training) ![alt text][image3]
* For zero bias filter parameter equals ~0.1 (last epoch of training) ![alt text][image4]

Here is the distribution of predicted data for the first track:



#### 4. Model architecture

I used the convolution neural network architecture from Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper suggested in the project video guide. 

Here is a visualization of the architecture (image from the mentioned paper):

![alt text][image1]

The following modifications have been added:
* I used ELU activation layers
* I added cropping layer in the model using a Keras Cropping2D layer (code line 13)
* I added data normalization in the model using a Keras Lambda layer (code line 16)
* I added Dropout layers after each fully connected layer (more about overfitting later)

#### 4. Training process

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.






#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
