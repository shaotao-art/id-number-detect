from numbers import Rational
import cv2
import numpy as np
# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img

# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
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

    # 剔除一些噪点
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints
# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））

def findBorderHistogram(path):
    list = []
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0] - vect_point[1]), (vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            list.append(border)
        list.sort(key=lambda range: range[0])
        for range in list[0:18]:
            borders.append(range[1:])
        # print(len(borders))
    return img, borders

# 切割结果并且修正分辨率
def getCut(img,borders):
    for i, border in enumerate(borders):
        newimg = np.zeros((32,32))
        imgCut = img[border[0][1]:border[1][1], border[0][0]:border[1][0]].copy()
        img_rows, img_cols = imgCut.shape[:2]
        resize_scale = np.max(imgCut.shape[:2])
        resized_img = cv2.resize(imgCut, (int(img_cols * 32 / resize_scale), int(img_rows * 32 / resize_scale)))
        img_rows, img_cols = resized_img.shape[:2]
        offset_rows = (32 - img_rows)//2
        offset_cols = (32 - img_cols)//2
        for x in range(img_rows):
            for y in range(img_cols):
                newimg[x+offset_rows, y+offset_cols] = resized_img[x,y]
        cv2.imwrite(r'./img{}.jpg'.format(i), newimg)

# 显示结果及边框
def showResults(path,borders):
    img = cv2.imread(path)
    # 绘制
    # print(img.shape)
    # print(borders)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        # if results:
        #     cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        #cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('test', img)
    cv2.imwrite(r'../image/imagelist/get.jpg',img)
    cv2.waitKey(0)

def gen_res(img, borders):
    res = []
    for i, border in enumerate(borders):
        newimg = np.zeros((32,32))
        imgCut = img[border[0][1]:border[1][1], border[0][0]:border[1][0]].copy()
        h, w = imgCut.shape
        ratio = h / 32
        new_w = int(w / ratio)
        imgCut = cv2.resize(imgCut, (new_w, 32))
        edge_w = (32 - new_w) // 2
        newimg[:, edge_w:edge_w + new_w] = imgCut
        newimg = 1 - newimg
        res.append(newimg)
    return res



path = 'TEST_IMG/out1.jpg'
img, borders = findBorderHistogram(path)
lst = gen_res(img, borders)
lst = [x.astype(np.int0) for x in lst]
print(lst[0])
from matplotlib import pyplot as plt
for i in range(16):
    plt.subplot(4, 16 // 4, i + 1)
    plt.imshow(lst[i], cmap="gray")
plt.show()
