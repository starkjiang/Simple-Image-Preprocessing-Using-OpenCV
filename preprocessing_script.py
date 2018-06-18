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
#def gb(img, size):
#    blur = cv2.GaussianBlur(img, (size, size), 0)
#    return blur

# check the images
if args["image"] == "invoice.tif":
    gray = mb(gray, 3)
    output = Morphology(gray, 2)
    cv2.imwrite("denoised_invoice.tif", output)
    cv2.imshow("Image", img)
    cv2.imshow("Output1",output)
    cv2.waitKey(0)

elif args["image"] == "GE.tif" or "Gibson.tif":
    output = mb(gray, 3)
    cv2.imwrite("denoised_{}".format(args["image"]), output)
    cv2.imshow("Image", img)
    cv2.imshow("Output2", output)
    cv2.waitKey(0)
"""
elif args["image"] == "Gibson.tif":
    output = gb(gray, 5) # for this image, Gaussian Blurring may be used
    cv2.imwrite("denoised_Gibson.tif", output)
    cv2.imshow("Image", img)
    cv2.imshow("Output3", output)
    cv2.waitKey(0)
"""
    
                    
