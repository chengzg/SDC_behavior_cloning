# **Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_summary.png "summary of the model"
[image2]: ./examples/outImage1.png "failure point 1"
[image3]: ./examples/outImage2.png "failure point 2"
[image4]: ./examples/outImage3.png "failure point 3"
[image5]: ./examples/inImage1.png "recovery image of failure point 1"
[image6]: ./examples/inImage2.png "recovery image of failure point 2"
[image7]: ./examples/inImage3.png "recovery image of failure point 3"
[image8]: ./examples/loss_over_epoch_graph.png "loss visualization"
[image9]: ./examples/center_recovery_1.jpg "recovery exampe 1"
[image10]: ./examples/center_recovery_2.jpg "recovery exampe 2"
[image11]: ./examples/center_recovery_3.jpg "recovery exampe 3"
[image12]: ./examples/center_recovery_4.jpg "recovery exampe 4"
[image13]: ./examples/center_recovery_5.jpg "recovery exampe 5"
[image14]: ./examples/center_recovery_6.jpg "recovery exampe 6"
[image15]: ./examples/center_recovery_7.jpg "recovery exampe 7"
[image16]: ./examples/center_recovery_8.jpg "recovery exampe 8"
[image17]: ./examples/center_recovery_9.jpg "recovery exampe 9"
[image18]: ./examples/center_recovery_10.jpg "recovery exampe 10"
[image19]: ./examples/center_recovery_11.jpg "recovery exampe 11"
[image20]: ./examples/center_recovery_12.jpg "recovery exampe 12"
[image21]: ./examples/center_recovery_13.jpg "recovery exampe 13"
[image22]: ./examples/center_recovery_14.jpg "recovery exampe 14"
[image23]: ./examples/center_recovery_15.jpg "recovery exampe 15"
[image24]: ./examples/center_recovery_16.jpg "recovery exampe 16"
[image25]: ./examples/flip_center_recovery_16.jpg "flip of recovery exampe 16"
[image26]: ./examples/right_1.jpg "right image sampe 1"
[image27]: ./examples/flip_right_1.jpg "flip of right image sample 1"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* image_util.py containing the script to load and generate the training and testing images
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
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

I used the Nvidia model architecture which contains
* 5 convolutional layers 

  |   layer name     |  kernel number  | kernel size |  stride | activation name    |
  | :--------------: | :-------------: |:-----------:| :------:| :-----------------:|
  | conv layer 1     | 24              | 5 x 5       | (2,2)   | relu |
  | conv layer 2     | 36              | 5 x 5       | (2,2)   | relu |
  | conv layer 3     | 48              | 5 x 5       | (2,2)   | relu |
  | conv layer 4     | 64              | 3 x 3       | (1,1)   | relu |
  | conv layer 5     | 64              | 3 x 3       | (1,1)   | relu |

* 1 flatten layer
   to Flatten the input

* 3 full connected feature layers and 1 output layer

  |   layer name     |  input number  | output number |  dropout |
  | :--------------: | :------------: |:-------------:| :-------:|
  |   fc layer 1     | 2112           | 100           | 0.2      |
  |   fc layer 2     | 100            | 50            | 0.2      |
  |   fc layer 3     | 50             | 10            | 0        |
  |   output layer   | 10             | 1             | 0        |

Before i do the acutal training, i do the following two pre-process steps:
* Normalize the input array
* Crop the original image with cropping=((70, 25), (0, 0)) to filter out noisy data

The model includes RELU layers to introduce nonlinearity 

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I directly take reference to the NVIDIA driving model for my model.

I thought this model might be appropriate because it is the architecture published by the autonomous vehicle team at NVIDIA

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with test_size=0.2. I directly introduce overfilt layer since the begining.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. 
Firstly the sample data provided in the original data set is used, I found along the track, there are 3 failures points as shown below:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To improve the driving behavior in these cases, i take another 2 laps by myself. One is the original direction and another lap is in reverse direction. And I take additional recovery driving  data from these failure points.  
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. And the corresponding new images at the given failure points are shown below:

![alt text][image5]
![alt text][image6]
![alt text][image7]

#### 2. Final Model Architecture

The final model architecture (model.py lines 27-57). Below is a summary of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Other than use the provided data, to capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn back to the center when they are offset to the sides.
These images show what a recovery looks like starting from right side of the lane :

![alt text][image11]
![alt text][image12]

![alt text][image13]
![alt text][image14]

![alt text][image15]
![alt text][image16]

![alt text][image17]
![alt text][image18]

![alt text][image19]
![alt text][image20]

![alt text][image21]
![alt text][image22]

![alt text][image23]
![alt text][image24]

I tried to record the video on Track two but the road condition is more complex. In the end, i did not include data from track 2.

To augment the data sat, I also flipped images and angles thinking that this would definitely since it is totally reasonal data for the model.
For example, here is an image that has then been flipped:

![alt text][image24]
![alt text][image25]

![alt text][image26]
![alt text][image27]
Etc ....

After the collection process, I had around 50,000 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs i tried was 3.
![alt text][image8]

I also tried to increase the epochs number to 5, but actually it does not help.  

#### 4. summary
* Given a powerful model architecture, the key point of success is the training dataset. Without providing a lot of training data, the addition or removal of training
data really affect the outcome of the whole model. 
* Using the left/right camera + flip is also key to success, so the network knows to recover back if is too far from the center
* As suggested by some other people from the forum, we can iterator a lot more epoches, save the model and retrain the model using only the recovery data
over the failure points. 
