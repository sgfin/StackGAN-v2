import os
import sys

import cv2
import numpy as np

def CLAHE(img, cliplimit=4.0, tilegridsize=(8,8)):
    """
        improve contrast to make bones more noticable in hand images
    """
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilegridsize)
    img = clahe.apply(img)
    return img

def histogram_equalization(img):
    """
        Convert min ~ max range to 0 ~ 255 range
    """

    img = img.astype(np.float64)

    min_val = np.min(img)
    max_val = np.max(img)
    img = img - min_val
    img = img/(max_val-min_val)

    img = img*255.0
    img = img.astype(np.uint8)

    return img

img_src_dir = "../data/xray_original/train/all"
img_dst_dir = "../data/xray_flip_hist/train/all"
#img_dst_dir = "../data/xray_resized_dim128/train/all"
if not os.path.exists(img_dst_dir):
    os.makedirs(img_dst_dir)

for img_file in os.listdir(img_src_dir):
    if not img_file.endswith(".png"):
        continue
    src_file_path = os.path.join(img_src_dir, img_file)
    dst_file_path = os.path.join(img_dst_dir, img_file)

    img = cv2.imread(src_file_path, 0)
    height, width = img.shape
    #new_height, new_width = (369, 370)
    new_height, new_width = (370, 370)
    #print height, width

    img_new = np.zeros((new_height, new_width), dtype=img.dtype)

    #left = (width - new_width)/2
    #top  = (height-new_height)/2
    #right = (width + new_width)/2
    #bottom = (height + new_height)/2
    #print top,bottom,left,right

    img_new = img[58:58+new_height, 143:143+new_width]
    #img_new = img[top:bottom, left:right]
    #img_new = cv2.resize(img_new, (128,128), interpolation=cv2.INTER_LINEAR)

    if "_right_" in img_file:
        img_new = cv2.flip(img_new, 1)

    #img_new = CLAHE(img_new)
    img_new = histogram_equalization(img_new)

    cv2.imwrite(dst_file_path, img_new)
