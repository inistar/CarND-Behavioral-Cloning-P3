# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn_arch.png "Model Visualization"
[image2]: ./examples/cente.jpg "Center"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flip.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* testing.ipynb to show any preprocess data

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 3 5x5 filter sizes and depths from 24, 36, 48 and 2 3x3 filter sizes with depths 64. The model at at the end has 3 fully connected layer with a dropout in between. (mode.py lines 127 - 137)

The model includes RELU layers to introduce nonlinearity (code line 127 - 131), and the data is normalized in the model using a Keras lambda layer (code line 125). 

The model uses adam has the optimizer and loss is measured using mean squared error. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 134). 

The model was trained and validated on 2 laps around the track. This means there was a balance amount of data for the car going to left and right.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).

The optimal batch_size is 32. When increasing the batch to 64 and 128, the model did not perform well. 

The image crop was also tuned to see if cutting more of the top portion of the image would help. But this did improve the model. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. I did not do much data collection of recovering the data from the left and right sides of the road in the simulator. This was done through data augmentation in `adjust_left_angle` and `adjust_right_angle` where data is generated to keep the vehicle in the center.

I also generated data in the simulator not through the left and right keys in the keyboard, but the using the mouse to capture the steering angle. This made the steering angle to be more clean then jumping from left or right back to center immediately. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car in the center of the lane in the simulator. The model would capture all the important aspects on the road to drive the car. 

My first step was to use a convolution neural network model similar to the LeNet. It was used as my base to see to how well the model was perform initially. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it takes in the RGB image and the input and normalizes the data. I also used dropouts to stop it from overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the sharp right corner to improve the driving behavior in these cases, Angle adjustment was set to 0.2 + some random number between 0-0.5. This helped with the recovery of the car back to the center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a NVIDIA convolution neural network with the following 3 5x5 filter sizes and depths from 24, 36, 48 and 2 3x3 filter sizes with depths 64. The model at at the end has 3 fully connected layer with a dropout in between. I thought this model might be appropriate because it was able to detect the road features automatically just through the steering angle. 

Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 2566 number of data points. I then preprocessed this data by adding left and right images with their randomly generated steering angle.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as it was peaking around that many epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
