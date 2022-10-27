import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

DEBUG = False

def read_img(path, desired = 80):
    """
    read img and resize img to short edge == desired
    """
    img = cv2.imread(path)
    w, h = img.shape[:2]
    min_len = min(w, h)
    scale_factor = desired / min_len
    r_w, r_h = int(scale_factor * w), int(scale_factor * h)
    img = cv2.resize(img, dsize = [r_h, r_w])
    return img

# 反相二值化图像
def accessBinary(img, threshold=128, erod_size = 3):
    """
    preprocess for split numbers:
    反相 -> thres -> erod     to get clean image
    """
    img = 255 - img

    _, thres = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)

    if erod_size > 0:
        kernel = np.ones((erod_size, erod_size), np.uint8)
        erod = cv2.erode(thres, kernel, iterations=1)

    if DEBUG == True:
        cv2.imshow("origin", img)
        cv2.waitKey()
        cv2.imshow('thres', thres)
        cv2.waitKey()
        cv2.imshow('erod', erod)
        cv2.waitKey()
    return erod


def extractPeek(array_vals, min_vals=10, min_rect=20):
    """
    extract peak in image histogram for spliting numbers
    """
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None


    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints


def findBorderHistogram(img):
    lst = []
    width_lst = []
    out = []
    img = accessBinary(img)
    # row scan
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # col scan
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            width_lst.append(vect_point[0] - vect_point[1])
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            lst.append(border)
        # choose max 18 for 18 number
    width_lst = np.array(width_lst)
    sorted_width = np.argsort(width_lst)[:18]
    for i in range(len(lst)):
        if i in sorted_width:
            out.append(lst[i])

    return img, out


def showResults(img, borders, save_res = False):
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
    
    cv2.imshow('show res', img)
    cv2.waitKey(0)
    if save_res == True:
        cv2.imwrite('./number_cut.jpg', img)
    


def gen_res(img, borders, save_path, out_size=28):
    """
    get each number in img accord to borders and save res to file
    """
    res = []
    for i, border in enumerate(borders):
        newimg = np.zeros((out_size, out_size))
        imgCut = img[border[0][1]:border[1][1], border[0][0]:border[1][0]].copy()
        h, w = imgCut.shape[:2]
        ratio = h / out_size
        new_w = int(w / ratio)
        imgCut = cv2.resize(imgCut, (new_w, out_size))
        edge_w = (out_size - new_w) // 2
        newimg[:, edge_w:edge_w + new_w] = imgCut
        if save_path is not None:
            number_path = os.path.join(save_path, f'img_{i}.jpg')
            print(f'saving number img to {number_path}')
            cv2.imwrite(number_path, newimg)
        res.append(newimg)
    return res


def get_each_number(path, save_path=None):
    """
    main function for this py file
    """
    img = read_img(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_, borders = findBorderHistogram(gray)
    if DEBUG == True:
        showResults(img, borders)
    
    
    img_to_gen_res = cv2.dilate(img_, np.ones((3, 3)))
    lst = gen_res(img_to_gen_res, borders, save_path)
    out = [x.astype(np.int0) for x in lst]
    return out

if __name__ == "__main__":
    path = './TEST_IMG/out4.jpg'
    out = get_each_number(path, save_path='.')
    print(f'getting {len(out)} numbers')
    if DEBUG == True:
        for i in range(16):
            plt.subplot(4, 16 // 4, i + 1)
            plt.imshow(out[i], cmap="gray")
        plt.show()
