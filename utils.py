import cv2
import numpy as np
def RBG_to_YUV(img):
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
    return img

def YUV_to_RGB(img):
    img = cv2.cvtColor( img, cv2.COLOR_YUV2RGB )
    return img

def RBG_to_dilation(img):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return 255-np.abs(dilation-erosion)
