# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]:  examples/center.jpg "Center Image"
[image2]:  examples/offRight.jpg "Off to the right"
[image3]:  examples/correction.jpg "Recovery Image"
[image4]:  examples/backToCenter.jpg "Back to center"
[image5]:  examples/nonFlipped.jpg "Original Image"
[image6]:  examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* run1.mp4 video recorded making a successful lap around track 1
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 filter sizes and  two 3X3 filter sizes with depths between 24 and 64 (model.py lines 63-67)

The model includes RELU activation layers to introduce nonlinearity (code lines 63-67), and the data is normalized in the model using a Keras lambda layer (code line 61) as well as Keras cropping layer (code line 62).

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers each with a dropout frequency of 0.5 in order to reduce overfitting (model.py lines 69 and 72).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (3 laps), reversing the direction with center lane driving (1 lap), recovering from the left and right sides of the road (1 lap) and gather more data around turns (about 2 laps). This gave me sufficient image scenarios to train the model.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple with smaller amounts of data and add layers as the I approached the desired output and added additional data.

My first step was to use a convolution neural network model similar to the NVIDIA Architecture. I thought this model might be appropriate because it has been tested and proven to work for computer vision type networks.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it included dropout layers.

Then I looking into image augmentation, adding more data along with trying variations to my model including max pooling and adding a fit generator in order to handle the additional data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track including most of the turns, the bridge, where there was water in the background and areas with no defined guard rail. To improve the driving behavior in these cases, I added more data focusing on the turns and the areas where the car seemed to have trouble as well as data augmentation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-81) consisted of a convolution neural network with the following layers and layer sizes:

 - Lamba function with input shape (160, 320, 3) and normalized image to x/255 - 0.5
 - Cropping layer taking 70px from the top and 25px from the bottom of the image
 - Convolution layer that is 5X5 and 24 deep with relu activation
 - Convolution layer that is 5X5 and 36 deep with relu activation
 - Convolution layer that is 3X3 and 48 deep with relu activation
 - Two convolution layers that are 3X3 and 64 deep with relu activation
 - Flatten function
 - Dropout of 0.5
 - Fully connected layer with 100 units
 - Fully connected layer with 50 units
 - Dropout of 0.5
 - Fully connected layer with 10 units
 - Output layer with 1 units
 - Compile layer with mean square errored loss function and 'adam' optimizer
 - Fit Generator with 5 epochs


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it gets too close to the edge. These images show what a recovery looks like starting from the right hand side :

![Off to the right][image2]
![Correcting][image3]
![Back to center][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would correct for overfitting left turns because the track is a counterclockwise loop. For example, here is an image that has then been flipped:

![Original Image][image5]
![Flipped Image][image6]


After the collection process, I had about 10k number of data points. I then preprocessed this data by normalizing it using a Keras lambda function and cropped the image by 70px (top) and 25px (bottom)


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss not decreasing after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

