# Face-Recognition-Face-Recognition

## Introduction

:dart: The purpose of this script is to test the `Python` library called `face-recognition` ([link](https://face-recognition.readthedocs.io/en/latest/face_recognition.html)).

With this library we can do the following:
* Face detection. [Tested]
* Face recognition. [Tested]
  * Know the distance of recognition. [Not tested]
* Draw landmarks. [Not tested]

## How does it work?

**1. Train**

We introduce an image to train the model and perform two steps.
1. Find the face in the photo.
2. Apply a model to extract the features of the face. By default, the model used to extract the facial features is the computer vision descriptor known as HOG, (Histogram of Oriented Gradients). However, the library also allows us to use a CNN model too. In this second case, it is recommended to have a GPU to run it.

In the following image, we can see how the detector finds Michael Scott's face and extract his face features.

![Training photo](https://github.com/mmr689/Face-Recognition-Face-Recognition/blob/main/results/train.png)

**2. Test**

Once we have trained our face recognition object. All we need to do is pass other photos, find the faces in them and see the confidence between the example face and the current face.

| ![Test 1 photo](https://github.com/mmr689/Face-Recognition-Face-Recognition/blob/main/results/test1.png) | ![Test 2 photo](https://github.com/mmr689/Face-Recognition-Face-Recognition/blob/main/results/test2.png) |
| --- | --- |

## Scripts

### Dependencies

* python==3.11.0
* opencv-contrib-python==4.7.0.72
* face-recognition==1.3.0

### main.py

Basic script that allows you to:

1. Upload a training photo and a set of test photos to recognise the trained face.
2. Use the computer camera to recognise faces. First we take a photo to define our face, then we take another photo and the program says if the training face is in the photo or not.
