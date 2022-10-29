import numpy as np
import cv2
from utils import _SHOW, gray_scale, DEBUG, contrast, thres_img, get_rotate_img_degree, rotate_img, erode



def read_img(file_path):
    """
    read img, resize to max_edge == 300
    """
    img = cv2.imread(file_path)
    print(f'reading img from {file_path}')
    w, h = img.shape[:2]
    max_len = max(w, h)
    desired = 300
    scale_factor = desired / max_len
    r_w, r_h = int(scale_factor * w), int(scale_factor * h)
    img = cv2.resize(img, dsize = (r_h, r_w))
    if DEBUG == True:
        _SHOW('resized img', img)

    print(f'origin img size {w, h}, resized img size {r_w, r_h}')  
    return img


def get_contours(img):
    """
    detect contours
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if DEBUG == True:
        tmp = np.zeros_like(img)
        res = cv2.drawContours(tmp, contours, -1, (250, 255, 255), 1)
        _SHOW('contours img', res)

    return contours


def anly_contours(img, contours):
    """
    find rectange in contours
    """
    print(f'find {len(contours)} contours')
    res_lst = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.03*cnt_len, True) #多边形逼近
        # if len(cnt) == 4 and cv2.isContourConvex(cnt) == True:
        if len(cnt) < 6 and cv2.isContourConvex(cnt) == True and cv2.contourArea(cnt) > 500:
            res_lst.append(cnt)
    
            if DEBUG == True:
                    print(len(cnt))
                    res = cv2.drawContours(img, [cnt], -1, (250, 250, 255), 1)
                    _SHOW(f'pic {len(cnt)}', res)

    print(f'get final {len(res_lst)} contours')
    return res_lst
    

def get_id_number_contour(cnt_lst, img):
    """
    get rectangle for id number according to w/h ratio
    """
    min_dist = 100

    w, h = img.shape[:2]
    res = None
    for i, cnt in enumerate(cnt_lst):
        num_p = len(cnt)
        # 先是列 再是行
        p_s = np.zeros((num_p, 2), dtype=np.int16)
        
        for i in range(num_p):
            p_s[i] = cnt[i]

        # left
        left = np.min(p_s[:, 0])
        right = np.max(p_s[:, 0])
        up = np.min(p_s[:, 1])
        down = np.max(p_s[:, 1])
        ratio = abs(right - left) / abs(down - up)
        if DEBUG == True:
            print(f"getting ratio {ratio}")
            cv2.rectangle(img, [left, up], [right, down], (111, 0, 255), 2)
            _SHOW('id number rectange', img)
        
        corner_points = [up/h, down/h, left/w, right/w]

        if abs(ratio - 9) < min_dist:
            min_dist = abs(ratio - 9)
            res = corner_points

    return res

def high_resolution_number(file_path, theta, ratio_lst, save_path):
    """
    get high resolution id number image from origin img
    """
    img = cv2.imread(file_path)
    img = rotate_img(img, theta)
    w, h = img.shape[:2]
    up, down, left, right = int(ratio_lst[0] * h), int(ratio_lst[1] * h), int(ratio_lst[2] * w), int(ratio_lst[3] * w)
    patch = img[up:down, left:right]
    cv2.imwrite(save_path, patch) 




def get_numbers(file_path, save_path, use_rotate = False):
    """
    main function for this py file
    """
    img = read_img(file_path)
    if use_rotate == True:
        theta = get_rotate_img_degree(img)
    else:
        theta = 0
    rotated_ = rotate_img(img, theta)
    gray_ = gray_scale(rotated_)
    contr_ = contrast(gray_)
    thres_1 = thres_img(contr_, thres=90)
    eroded_img = erode(thres_1, kernel_size=9)
    binary_img = thres_img(eroded_img, thres=128)

    contours = get_contours(binary_img)
    rotated_img = rotate_img(img, theta)
    cnt_lst = anly_contours(rotated_img, contours)
    ratio_lst = get_id_number_contour(cnt_lst, rotated_img)
    high_resolution_number(file_path, theta, ratio_lst, save_path)


if __name__ == "__main__": 
    get_numbers('./TEST_IMG/test1.jpg', './out1.jpg')
    # get_numbers('./TEST_IMG/test2.jpg', './out2.jpg')
    # get_numbers('./TEST_IMG/test3.jpg', './out3.jpg')
    # get_numbers('./TEST_IMG/test4.jpg', './out4.jpg')
    # get_numbers('./TEST_IMG/test5.jpg', './out5.jpg')
    # get_numbers('./TEST_IMG/test6.jpg', './out6.jpg')
