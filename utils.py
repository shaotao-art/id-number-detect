import cv2
import numpy as np
import math

DEBUG = False

def _SHOW(win_name, img):
    cv2.imshow(win_name, img)
    cv2.waitKey()


def gray_scale(img):
    """
    grayscale
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    if DEBUG == True:
        _SHOW('gray', gray) 
    return gray 


def blur(img):
    blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1)
    if DEBUG == True:
        _SHOW('blur', blur)
    return blur

def contrast(img):
    clahe = cv2.createCLAHE(3, (8, 8))
    dst = clahe.apply(img)
    res = cv2.convertScaleAbs(dst, alpha = 1.7, beta = 0)
    if DEBUG == True:
        _SHOW("contrast", res)
    return res

def thres_img(img, thres = 128):
    """
    threshold img to get binary image
    """
    _, thresh = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    
    if DEBUG == True:
        _SHOW('binary img', thresh)

    return thresh

def inverse(img):
    img = 255 - img
    if DEBUG == True:
        _SHOW("fanxiang", img)
    return img


def erode(img, kernel_size):
    erod = cv2.erode(img, np.ones((kernel_size, kernel_size)))
    if DEBUG == True:
        _SHOW("fushi", erod)
    return erod

def rotate_img(image, angle, center = None, scale = 1.0):
    """
    rotate an img
    """
    w, h = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)   
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale) 
    out = cv2.warpAffine(image, wrapMat, (h * 2, w * 2))

    if DEBUG == True:
        _SHOW('rotated img', out)
    return out


def get_rotate_img_degree(img):
    """
    compute rotate degree for horized img
    """
    canny = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 100, maxLineGap = 10)
    
    if DEBUG == True:
        _SHOW('canny', canny)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if DEBUG == True:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if DEBUG == True:
        _SHOW('hough', img)

    k = float(y1 - y2) / (x1 - x2)
    theta = np.degrees(math.atan(k))
    return theta
