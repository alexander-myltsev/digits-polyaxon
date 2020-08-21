# Digits

```info
author: M Borisyak, hypothetical V. Belavin 
```

## Introduction

> Here we pretend it is 2005 or something like that.

Recently proposed Convolutional Neural Networks (CNN) [lecun]
show great results for handwritten-digit recognition.
Here, we propose to apply CNN for recognizing house numbers.

[lecun]: LeCun Y, Boser B, Denker JS, Henderson D, Howard RE, Hubbard W, Jackel LD. Backpropagation applied to handwritten zip code recognition. Neural computation. 1989 Dec;1(4):541-51.

## Motivation

Convolutional Neural Networks offer a number advantages for image recognition problems:
- its hierarchical structure nicely lays on actual hierarchical structure of natural images;
- shift-"invariance" is definitely present in natural images;
- they are highly versatile --- one can adjust number of kernels, kernel sizes, pooling layers etc;

Of course, not all properties of natural images are also present in CNNs like rotational invariance and scale invariance:
one can compensate for this by increasing number of filters and most importantly --- data augmentation.

Here, we propose to apply CNNs to street signs, specifically, to house numbers.
Hopefully, similarity with MNIST allows the success of CNN architectures on MNIST to be transferred on
our problem of recognising house numbers.

## Dataset

*(copied from http://ufldl.stanford.edu/housenumbers/)*

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal
requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST
(e.g., the images are of small cropped digits),
but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved,
real world problem (recognizing digits and numbers in natural scene images).
SVHN is obtained from house numbers in Google Street View images.

Additionally, we suggest to use MNIST for monitoring.

## Suggested models

A number of CNN architectures can be applied for this task:
- logistic regression as a simple baseline;
- simple CNN architecture (VGG-like);
- ResNet-like (ignore hypothetical time travel);

## Research plan

1. implement proposed models;
2. test them on MNIST and SVHN;
3. compare.
4. ...
5. profit!

## Authorship policy

Authorship policy is the default one.