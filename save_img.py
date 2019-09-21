# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:58:57 2019

@author: yilis
"""

import os 
import cv2  
import numpy as np
import matplotlib.pyplot as plt 
#cap = cv2.VideoCapture("F:\\traffic\\sun\\q1.avi") #读入视频文件

def save_img(cap, save_path, count,start,bgr=True):   
    '''
    for save img from existed rain/sun avi bgr video
    cap: video loaded by Opencv
    save_path:folder for saving img
    count:save a image between the number of frames
    start:save image name
    output: images saved in 'save_path' folder
    '''
#    flag = 0
    c=1
    while (cap.isOpened()):
        ret,im = cap.read()#获取图像
        if not ret:
            break
        if bgr:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = rgb
        if (c%count == 0):
            cv2.imwrite( save_path + str(start) + ".jpg",im)#保存图片
            start+=1
        c += 1
    print('number of pictures:', start)
    
def save_img_time(cap,save_path,lenth,rgb=False):
    c=1     
    if cap.isOpened(): #判断是否正常打开
        rval , frame = cap.read()
    else:
        rval = False
    for i in range(lenth):     
        if rval:   #循环读取视频帧
            rval, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                frame = rgb
    #        if(c%timeF == 0): #每隔timeF帧进行存储操作
            cv2.imwrite(save_path + str(c) + '.jpg',rge) #存储为图像
            c = c + 1
            cv2.waitKey(1)
    cap.release()

def find_boundary(img):
    '''
    for self recorded images, to find the IOU boundaries.
    inout: RGB images
    output: array with (xmin, ymin, height, width)
    '''
#    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binaryImg = cv2.Canny(gray,100,250)
#    cv2.imshow('mask',binaryImg)
    h = cv2.findContours(binaryImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = h[0]
#    img_copy = img.copy()
    w0 = 0
    h0 = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
#        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
        if w > w0 and h > h0:
            x1 = x
            y1 = 0
            w1 = w
            h1 = y+h
        w0,h0 = w,h    
#    cv2.rectangle(img_copy,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
#    cv2.imshow('contours(green)', img_copy)
#    cv2.waitKey(0)
#    cv2.destoryALLWindows()  
    return np.array([x1,y1,w1,h1])


def find_bbox(img, rgb = True):
    '''
    just to recogenize the annotation box after object detection 
    might need to change the color array when the result box color is different than mine
    the color array should be in hsv coordinate
    box (0,255,255) yellow
    input:image after detection
    output:smallest bounding box which include all detected object
    '''
    if rgb:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    img_copy = img.copy()
#    lower_color = color_array - np.array([10,10,10])
#    upper_color = color_array + np.array([10,10,10])
#  array([ 55, 152, 151], dtype=uint8)
#        array([ 54, 112, 116], dtype=uint8)
    lower_color =  np.array([40,110,110])#might need change here
    upper_color = np.array([64,160,160])#might need change here
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
#    cv2.imshow('mask',mask)
    h = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = h[0]
    box = np.zeros((len(contours), 4),dtype = np.int)
    i = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
    #    print(x,y,w,h)
        box[i,:] = x,y,x+w,y+h
#        print(x,y,w,h)
        i+=1
#        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
#        cv2.imshow(str(i), img_copy)
    xmin = np.min(box[:,0])
    ymin = np.min(box[:,1])
    wmax = np.max(box[:,2]) - xmin
    hmax = np.max(box[:,3]) - ymin
#    print(box,xmin,ymin)
#    cv2.rectangle(img_copy,(xmin,ymin),(xmin+wmax,ymin+hmax),(0,255,255),2)
#    cv2.imshow('boundary', img_copy)
#    cv2.waitKey(0)
#    cv2.destoryALLWindows()  
#    if find_bound:
#        mask2 = cv2.inRange(hsv_img, np.array([35,0,250]), np.array([50,8,255]))
#        h2 = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#        contours2 = h[0]
#        
#        return np.array([])
    return np.array([xmin,ymin,wmax,hmax])

def draw_ROI(coor,img,color,path):
    '''
    draw ROI area in public based images, this function CANNOT segment images, just test
    coor:location array from find_bbox 
    img:origional image
    color:ROI area color
    path: new image save path
    '''
    x,y,w,h = coor
    a,b = np.shape(img[:,:,0])
    if w >= a or h >= b:
        assert "ROI range extend the images"
    if x<0:
        x = 0
    if y<0:
        y = 0
    if x+w >= a:
        x = a-w-1 
    if y+h >= b:
        y = b-h-1
    cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
    font = cv2.FONT_HERSHEY_PLAIN 
    #cv2.putText(img,label1,(x,y+h+17), font, 1, color,1)
    #cv2.putText(img,label2,(x,y+h+34), font, 1, color,1)
    cv2.imwrite(path, img)

def find_ratio(path):
    
    print('finding ratio...')
    tiny_rec = []
    large_rec = [] 
    w_ratio = [] 
    h_ratio = []
    for i in range(200):
        filename = path + 'a'+ str(i+1) + '.jpg'
        print(filename)
        img = cv2.imread(filename)
#        color = np.array([0,0,0])
        tiny_rec.append(find_bbox(img))
        large_rec.append(find_boundary(img))
        ratio1 = large_rec[-1]/tiny_rec[-1]
        w_ratio.append(ratio1[2])
        h_ratio.append(ratio1[3]) 
        wr_max = np.max(w_ratio)
        hr_max = np.max(h_ratio) 
        wr_min = np.min(w_ratio)
        hr_min = np.min(h_ratio) 
        wr_avg = np.mean(w_ratio)
        hr_avg = np.mean(h_ratio)
    print('finish finding')
    return tiny_rec, large_rec, np.array([wr_max, wr_min, wr_avg]), np.array([hr_max, hr_min, hr_avg])    

if __name__ == '__main__':
#    path = './roitest/'
#    tiny_rec, large_rec, w, h = find_ratio(path)
#    bound_arry = np.array(large_rec)
    #new_ratio = []
    #for i in range(200):
    #    img = cv2.imread(path+'a'+str(i+1)+'.jpg')
    #    img_copy = img.copy()
    #    sx,sy,sw,sh = tiny_rec[i]
    #    neww = sw*w[-1]
    #    newh = sh*h[-1]
    #    new_roi = np.array([sx+0.5*sw-0.5*neww,sy+0.5*sh-0.5*newh,neww,newh]).astype(int)
    #    rt = new_roi[-2:]/large_rec[i][-2:]
    #    new_ratio.append(rt)
    #    lebel1 = "generated ROI area"
    #    lebel2 = "new_w/old_w:" + "%.3f"%rt[0] +", "+ "new_h/old_h:" + "%.3f"%rt[1]
    #    draw_ROI(new_roi,img_copy,(0,255,255),lebel1,lebel2,'./roigene/'+str(i+1)+'.jpg')
        
    #save_img_time(cap, "F:\\traffic\\ROI_test\\", 200)   
        
    #img = cv2.imread('t2.png')
    #img_copy = img.copy()
    #bx,by,bw,bh = find_boundary(img)
    #x,y,w,h = find_bbox(img, np.array([60,218,173]))
    #rat_w = bw / w
    #rat_h = bh / h
    #off_x = bx - x + 0.5*(bw-w)
    #off_y = by - y + 0.5*(bh-h)
    
    #cv2.rectangle(img_copy,(bx,by),(bx+bw,by+bh),(0,0,255),2)
    #cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,255),2)
    #cv2.imshow('boundary', img_copy)
    #cv2.waitKey(0)
    #cv2.destoryALLWindows()
    #list_w = [a[0] for a in new_ratio]
    #list_h = [b[1] for b in new_ratio]
    #plt.plot(np.arange(200),[a[0] for a in new_ratio],c = 'red', label = "self generated w/previous w")
    #plt.plot(np.arange(200),[b[1] for b in new_ratio],c = 'blue', label = "self generated h/previous h")
    #axes = plt.gca()
    #axes.set_ylim([0,2])
    #plt.legend(loc='best')
    cap = cv2.VideoCapture("rain/r5.avi") #读入视频文件
    save_img(cap, './test_img2/', 1,4000, bgr = True)#q1-4, r4,2  record1-5
