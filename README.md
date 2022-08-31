# Yoga Pose Detection

## Overview
Human activity recognition is one of important problems in computer vision.
It includes accurate identification of activity being performed by a person or group of people. 


In this project, we will be classifying yoga poses into different classes. 
We have 19 types of Asanas in our dataset and 29K images for
training a machine learning model. 

Yoga pose estimation has multiple applications such as in creating a
mobile application for yoga trainer.

This is a kaggle competition and link of competition is : <a href = "https://www.kaggle.com/t/4723dbdfc94cf9183c664a32e1a2773f"> Link </a>

The competition ran for 1 month and we need to make submissions on weekly basis.

## Tasks

We are given around 29K training images with 19 types of Asanas and the images are taken from 4 different camera angles. The camera 
angles of training and test images are different i.e. training images
are taken from 3 camera angles and test images are taken from the 
fourth camera angle. We need to predict the test data yoga poses 
using Machine Learning. 

## Techniques Used

I have used 2 techniques to predict Yoga poses:

### Convolution Neural Networks (CNN)

I have built a simple <a href = "https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53" > 
CNN </a> model to predict yoga poses. Initially, I
was getting accuracy of 35% on test data. I analyse the images and 
use appropriate data augmentation techniques to improve the accuracy
from 35% to 76% using the same model. The data augmentation techniques
that I have used are:

1. <a href = "https://pytorch.org/vision/0.9/transforms.html" > Padding </a>
2. <a href = "https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html"> HorizontalFlip </a> with Padding
3. <a href = "https://pytorch.org/vision/0.9/transforms.html" > RandomPerspective </a> with Padding
4. HorizontalFlip with RandomPerspective and Padding

The detailed analysis could be found at <a href = "Report.pdf"> Report.pdf </a>
under Week 1 section. 

### PoseNet

After analysing the images, I observed that if we remove background
and only focus on the human body, then for different camera angles,
the body is at 90 degree angle i.e. it is like body is rotating 
about the y-axis if we assume the image is in X-Y plane. So, if we
could find the coordinates of different parts of body, then we can use 
neural network to learn from the coordinates and the features that
we will require are X,Y,Z,X<sup>2</sup>,Z<sup>2</sup>. So, I used 
<a href = "https://arxiv.org/abs/1803.08225"> PoseNet </a> to find X-Y coordinates of 17 key points of the body. But the
problem is we need to find the Z coordinate. For that I calculated
the bone's length of human body and then I calculated z-coordinate
using that. 

The detailed analysis could be found at <a href = "Report.pdf"> Report.pdf </a>
under Week 2 section. 






