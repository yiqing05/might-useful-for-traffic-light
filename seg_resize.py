# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:25:22 2019

@author: yilis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 
import os
from PIL import Image
from save_img import*

def seg(load_path,save_path):
    i = 1
    for root,dirs,files in os.walk(load_path):
        for name in files:
#            print(os.path.join(root,name))
            img_path = os.path.join(root,name)
            im = cv2.imread(img_path)
            img = Image.open(img_path)
            boundary = find_boundary(im)
            cropped = img.crop(boundary)
            cropped.save(save_path + name)
            i += 1
    print('finish %d images segmentation' % (i))
def resize(load_path,save_path,size = 128, ex_boundary = True):
    i = 0
    for root,dirs,files in os.walk(load_path):
        for name in files:
            i += 1
#            print(os.path.join(root,name))
            img_path = os.path.join(root,name)
            im = cv2.imread(img_path)
#            img = Image.open(img_path)
#            print(img_path)
            img = im.copy()
            if ex_boundary:
                boundary = find_boundary(im)
            else:
                boundary = find_bbox(im)
            x,y,w,h = boundary
            cropped = img[y:y+h, x:x+w]
#            cv2.imwrite(save_path+name, cropped)
            scale = max(w,h)/float(size)
            new_w, new_h = int(w/scale), int(h/scale)
            resize_img = cv2.resize(cropped, (new_w,new_h))
#            cv2.imwrite(save_path+name, resize_img)
#            print(resize_img)
###         0 padding
            if new_w >= new_h:
                top = int((size-new_h)/2)
                bottom = size - top - new_h
                left = 0
                right = 0
            elif new_w < new_h:
                top = 0
                bottom = 0
                left = int((size-new_w)/2)
                right = size - left - new_w
#            else:
#                top,bottom,left,right = 0,0,0,0
            pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            cv2.imwrite(save_path + name, pad_img)
    print('finish %d images resizing' % (i))

def resize_only(load_path, save_path,size=128):
    i = 0
    for root,dirs,files in os.walk(load_path):
        for name in files:
            i += 1
#            print(os.path.join(root,name))
            img_path = os.path.join(root,name)
            im = cv2.imread(img_path)
#            img = Image.open(img_path)
            img = im.copy()
            h,w = img.shape[:2]
            scale = max(w,h)/float(size)
            new_w, new_h = int(w/scale), int(h/scale)
            resize_img = cv2.resize(img, (new_w,new_h))
            if new_w >= new_h:
                top = int((size-new_h)/2)
                bottom = size - top - new_h
                left = 0
                right = 0
            elif new_w < new_h:
                top = 0
                bottom = 0
                left = int((size-new_w)/2)
                right = size - left - new_w
#            else:
#                top,bottom,left,right = 0,0,0,0
            pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            na = name[:-4]+'.jpg'
#            print(na)
            cv2.imwrite(save_path+na, pad_img)
    print('finish %d images resizing' % (i))            
            
if __name__ == '__main__':            
#    resize_only('./record_img/','./record_resize/',size=128)
    resize('./test_img2/','./samples4/')       
        
    
        