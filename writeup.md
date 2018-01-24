# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md... it's this.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is directly based off of the Nvidia architecture discribed in [this paper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (train.py lines 74). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (the split is located at train.py 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 82).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  Originally I had recorded about 15 laps in the center of the track, however, this data had an issue where it would turn off on the same spot. So after quite a bit more testing, I ended up training it on only about 3 laps driving around and a few "coming back to the center from the edge" rocordings around the areas where the network had issues.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Don't fix what ain't broke, thanks nvidia!  (I would have liked to spend more time messing around with the network, but since I'm already a few weeks behind I decided I'd just get it to a state that worked as fast as possible).

#### 2. Final Model Architecture

From the Nvidia paper mentioned above:

![alt text][image1]
_(Source: https://devblogs.nvidia.com/deep-learning-self-driving-cars/)_
#### 3. Creation of the Training Set & Training Process

The dateset was recorded as mentiod before & was augmented by randomly picking one of the images out of the center, left & right images, changing the left & right steer\_angle by 0.2 or -0.2 respectivly.  Then, randomly I flipped the image.  One interesting thing about this data augmentation is it led to the training accuracy being worse then the validation accuracy in a few cases.
