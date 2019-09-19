# -*- coding: utf-8 -*-
import cv2
import os
import glob
def imgs2video(imgs_dir, save_name):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (128, 128))
#    imgs = glob.glob(os.path.join(imgs_dir, '*.jpg'))
#    for i in range(1,len(imgs)+1): 
    for root,dirs,files in os.walk(imgs_dir):
        for name in files:
            imgname = os.path.join(root,name) 
#            print(imgname)
            frame = cv2.imread(imgname) 
            video_writer.write(frame)
        video_writer.release()
    
    
imgs2video('./imgoutrain/','testrain_result.avi')
