"""
This code is for image preprocessing to reduce the noise of images
Required installed softwares:
Python 3, OpenCV, Numpy
"""

# import necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Morphological transformation
def Morphology(img, size):
    kernel = np.ones((size, size), np.uint8)
    mor_tran = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return mor_tran

# median blurring
def mb(img, size):
    blur = cv2.medianBlur(img, size)
    return blur

# Gaussian blurring
def gb(img, size):
    blur = cv2.GaussianBlur(img, (size, size), 0)
    return blur

# Grayscale denoising using non-local means denoising algorithm
def nmda(img, h, size1, size2):
    dst = cv2.fastNlMeansDenoising(img, None, h, size1, size2)
    return dst

# mask
def Mor_mask(img, gray, size1, size2):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (size1, size1))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (size2, size2))
    mask = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255
    output = img * mask
    return output

# bilateral filter
def bil(img, filter_size, sigma):
    blur = cv2.bilateralFilter(img,filter_size,sigma,sigma)
    return blur

# check different images based on the args
if args["image"] == "invoice.tif":
    #gray = bil(gray, 9, 75)
    gray = gb(gray, 3)
    #gray = Morphology(gray, 2)
    #output = nmda(gray, 10, 7, 21)
    #output = Mor_mask(img, gray, 5, 2)
    output = bil(gray, 5, 75)
    cv2.imwrite("denoised_{}".format(args["image"]), output)
    cv2.imshow("Image", img)
    cv2.imshow("Output1",output)
    cv2.waitKey(0)

elif args["image"] == "GE.tif":
    output = mb(gray, 3)
    cv2.imwrite("denoised_{}".format(args["image"]), output)
    cv2.imshow("Image", img)
    cv2.imshow("Output2", output)
    cv2.waitKey(0)

elif args["image"] == "Gibson.tif":
    output = gb(gray, 5)
    cv2.imwrite("denoised_{}".format(args["image"]), output)
    cv2.imshow("Image", img)
    cv2.imshow("Output3", output)
    cv2.waitKey(0)
    
                    
