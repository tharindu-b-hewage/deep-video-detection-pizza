# deep-video-detection-pizza

## Introduction

Object detection is a well known task in the field of machine vision. So far it has been done using conventional image filters. Since this method could be applied for a wide range of applications, finding the suitable and accurate filter was always a challenge. 
  In this project we try to solve the problem of detecting a pizza image in a live video stream without using handcrafted features but using state of the art convolutional neural network running on real time.

  ## Architecture

  This system first extract incoming video frame. Then run a sliding window on it to capture several cropped snaps on several part in the image. Then scale all of them to the same size and stack them together to create a batch of images to feed in to a covolutional neural network(Alex net architecture) in a single pass. Then the confident score will be checked in each photo untill it passes a certain threshold to conclude existance of a pizza. This conclution process would be done through consecutive set of images(fixed number which could be changed with a configuration file) to get a correct judgement of the existance of a pizza image through temporal axis.

  ## How to use

  This folder include a visual studio project to build a binary to classify a given video file. A pre-built binary would be included in the release folder. In order to run the project, a trained caffe model is required. Which could be obtained by https://drive.google.com/open?id=0B94z5dLZzA5nSWdUVlNfZmJIUkU.
   Due to the size limitations, only the required C++ files are included. This project was built with:
   * MS visual studio 15
   * Caffe
   * opencv

## Running a TEST

Download release folder in https://drive.google.com/open?id=0B94z5dLZzA5nTlROOThWelJkVms. Download model files and copy them in to release folder. Modify config file in the same location to include testing video and the type of hardware used. Run Pizza New 2.exe file.


## Important notice

Any use of this project or replication for commercial purpose is strictly prohibited.
