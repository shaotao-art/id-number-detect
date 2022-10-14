import numpy as np
import cv2
import math
# i am answer why we did not solve hard situation, beacause we can get a high quality id card image without perpective disorder by design a good fontend to capture a good iamge

DEBUG = True


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
        cv2.imshow('rotated img', out)
        cv2.waitKey()
    return out


def read_img(file_path):
    """
    read img, resize to max_edge == 300
    """
    #读取图片，灰度化
    img = cv2.imread(file_path)
    print(f'reading img from {file_path}')
    w, h = img.shape[:2]
    max_len = max(w, h)
    desired = 300
    scale_factor = desired / max_len
    r_w, r_h = int(scale_factor * w), int(scale_factor * h)
    img = cv2.resize(img, dsize = [r_h, r_w])
    if DEBUG == True:
        cv2.imshow('resized img', img)
        cv2.waitKey() 
    print(f'origin img size {w, h}, resized img size {r_w, r_h}')  
    return img


def gray_scale(img):
    """
    grayscale
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    return gray 


def erod(gray, kernel_size = 7):
    """
    erode img
    """
    #腐蚀、膨胀
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    
    if DEBUG == True:
        cv2.imshow('erode', erode_Img)
        cv2.waitKey()
    return erode_Img


def thres_img(img, thres = 100):
    """
    threshold img to get binary image
    """
    _, thresh = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    
    if DEBUG == True:
        cv2.imshow('binary img', thresh)
        cv2.waitKey()
    return thresh


def get_contours(img):
    """
    detect contours
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if DEBUG == True:
        tmp = np.zeros_like(img)
        res = cv2.drawContours(tmp, contours, -1, (250, 255, 255), 1)
        cv2.imshow('contours img', res)
        cv2.waitKey()
    return contours


def anly_contours(img, contours):
    """
    find rectange in contours
    """
    print(f'find {len(contours)} contours')
    res_lst = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #多边形逼近
        # if len(cnt) == 4 and cv2.isContourConvex(cnt) == True:
        if len(cnt) == 4:
            res_lst.append(cnt)
    
    if DEBUG == True:
        for cnt in res_lst:
            res = cv2.drawContours(img, [cnt], -1, (250, 250, 255), 1)
        cv2.imshow('processed contours img', res)
        cv2.waitKey()
    print(f'get final {len(res_lst)} contours')
    return res_lst
    

def get_id_number_contour(cnt_lst, img):
    """
    get rectangle for id number according to w/h ratio
    """
    max_ratio = -1

    w, h = img.shape[:2]
    lst = []
    for i, cnt in enumerate(cnt_lst):
        # 先是列 再是行
        p_s = np.zeros((4, 2), dtype=np.int16)
        
        p0 = cnt[0][0]
        p1 = cnt[1][0]
        p2 = cnt[2][0]
        p3 = cnt[3][0]
        p_s[0] = p0
        p_s[1] = p1
        p_s[2] = p2
        p_s[3] = p3
        
        # left
        left = np.min(p_s[:, 0])
        right = np.max(p_s[:, 0])
        up = np.min(p_s[:, 1])
        down = np.max(p_s[:, 1])
        ratio = abs(right - left) / abs(down - up)

        lst.append([up/h, down/h, left/w, right/w])

        if ratio > max_ratio:
            max_ratio = ratio
            max_idx = i

    if DEBUG == True:
        up, down, left, right = int(lst[max_idx][0] * h), int(lst[max_idx][1] * h), int(lst[max_idx][2] * w), int(lst[max_idx][3] * w)
        cv2.rectangle(img, [left, up], [right, down], (111, 0, 255), 2)
        cv2.imshow('id number rectange', img)
        cv2.waitKey()
    return lst[max_idx]



def high_resolution_number(file_path, theta, ratio_lst, save_path = 'final_number.jpg'):
    """
    get high resolution id number image from origin img
    """
    img = cv2.imread(file_path)
    img = rotate_img(img, theta)
    w, h = img.shape[:2]
    up, down, left, right = int(ratio_lst[0] * h), int(ratio_lst[1] * h), int(ratio_lst[2] * w), int(ratio_lst[3] * w)
    patch = img[up:down, left:right]
    cv2.imwrite(save_path, patch) 



def get_rotate_img_degree(img):
    """
    compute rotate degree for horized img
    """
    #边缘检测
    canny = cv2.Canny(img, 50, 150)
    #霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 100, maxLineGap = 10)
    
    if DEBUG == True:
        cv2.imshow('canny', canny)
        cv2.waitKey()
    #画出线条
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if DEBUG == True:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if DEBUG == True:
        cv2.imshow('hough', img)
        cv2.waitKey()

    # 计算角度,因为x轴向右，y轴向下，所有计算的斜率是常规下斜率的相反数，我们就用这个斜率（旋转角度）进行旋转
    k = float(y1 - y2) / (x1 - x2)
    theta = np.degrees(math.atan(k))
    return theta

def main(file_path, save_path, use_rotate = False):
    img = read_img(file_path)
    gray = gray_scale(img)
    if use_rotate == True:
        theta = get_rotate_img_degree(gray)
    else:
        theta = 0
    gray = rotate_img(gray, theta)
    rotated_img = rotate_img(img, theta)
    eroded_img = erod(gray)
    binary_img = thres_img(eroded_img)
    contours = get_contours(binary_img)
    cnt_lst = anly_contours(rotated_img, contours)
    ratio_lst = get_id_number_contour(cnt_lst, rotated_img)
    high_resolution_number(file_path, theta, ratio_lst, save_path)


if __name__ == "__main__": 
    main('test6.jpg', 'out6.jpg')
