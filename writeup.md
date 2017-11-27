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

[image11]: ./visualisations/loss.png "Loss"


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

---
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

#### 4. Analyzing post augmented data

Here is the final data distribution:

* For zero bias filter parameter equals 1 (first epoch of training) ![alt text][image3]
* For zero bias filter parameter equals ~0.1 (last epoch of training) ![alt text][image4]

As we can see we managed to achieve appropriate data distribution for training purposes.

#### 5. Model architecture

I used the convolution neural network architecture from Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper suggested in the project video guide. 

Here is a visualization of the architecture (image from the mentioned paper):

![alt text][image1]

The following modifications have been added:
* I used ELU activation layers
* I added cropping layer in the model using a Keras Cropping2D layer (code line 13)
* I added data normalization in the model using a Keras Lambda layer (code line 16)
* Dropout layers

Since we heavily augmented the data we can expect our model to generalize well already. But I also added Dropout layers after each fully connected layer to adrees possible overfitting.

#### 6. Training process

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used mean square error loss and also mean absolute error metrics for both training and validation set to monitor training performance. The model was trained using Adam optimizer.

![alt text][image11]

#### 7. Analizing model performance

Here is the distribution of predicted data for the first track:

![alt text][image5]

And finally here are raw, augmented and predicted data distributions combines:

![alt text][image6]

The model drives the first track well on the speeds up to 20 mph. The video of the run is in **run1.mp4** file. 
This model also drives the second track decently. It only went bumped into the wall once and once the speed controller stoped it, so I had to manualy make it move again. Note that this was achieved without any data from the second track. The video of the second track run is in **run2.mp4** file.

I also tried different models. I managed to ran first track on the max speed of 30mph with the model that had high zero bias filter parameter (thus having more smooth data). However this model required some changes in drive.py. I had to add multiplier to predicted angle, because model didn't manage to drive sharp corners without it. I thought we are not allowed to do this, so I didn't use this approach in a final model.

