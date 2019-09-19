# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:35:41 2019

@author: yilis
"""
import copy
from lxml.etree import Element, SubElement, tostring, ElementTree
import os

def transfer(image_dir,xml_ori,txt_dir,xml_save,name):
    with open(txt_dir) as f:
        trainfiles = f.readlines()  # 标注数据 格式(filename label x_min y_min x_max y_max)
    
#    num = len(trainfiles)
    tree = ElementTree()
    label = 'light'
    xml_file = name.replace('jpg', 'xml')
#    tree.parse(xml_dir)
#    root.find('filename').text = name
#    sz = root.find('size')
#    sz.find('height').text = str(128)
#    sz.find('width').text = str(128)
#    sz.find('depth').text = str(3)
    flag = True
    for line in trainfiles:
        trainFile = line.split()
        

#        file_names.append(file_name)
#        lable = 'light'
        xmin = trainFile[1]
        ymin = trainFile[2]
        xmax = trainFile[3]
        ymax = trainFile[4]
        if flag:
            tree.parse(xml_ori)
            root = tree.getroot()
            root.find('filename').text = name
            root.find('path').text = image_dir
            sz = root.find('size')
            sz.find('height').text = str(128)
            sz.find('width').text = str(128)
            sz.find('depth').text = str(3)
            obj = root.find('object')
            obj.find('name').text = label
            bb = obj.find('bndbox')
            bb.find('xmin').text = xmin
            bb.find('ymin').text = ymin
            bb.find('xmax').text = xmax
            bb.find('ymax').text = ymax
            flag = False
        else:
            tree.parse(xml_save + xml_file)
            root = tree.getroot()
            obj_ori = root.find('object')
            obj = copy.deepcopy(obj_ori)
            obj.find('name').text = label
            bb = obj.find('bndbox')
            bb.find('xmin').text = xmin
            bb.find('ymin').text = ymin
            bb.find('xmax').text = xmax
            bb.find('ymax').text = ymax
            root.append(obj)
     
#        xml_file = name.replace('jpg', 'xml')
#        print(xml_save,xml_file)
        tree.write(xml_save + xml_file, encoding='utf-8')

def total_trans(load_path,save_path,image_path,xml_ori):
    i = 0
    for root,dirs,files in os.walk(load_path):
        for name in files:
            i+=1
            txt_path = os.path.join(root,name[:-4]+'.txt')
            img_path = image_path + name
            img_name = name[:-4] + '.jpg'
#            print('txt_path',txt_path)
#            print('img_path',img_path)
#            print('img_name', img_name)
            transfer(img_path,xml_ori,txt_path,save_path,img_name)
    print('finish %d files transfer to xml' % (i))

if __name__ == '__main__':

    image_folder = 'record_resize/'
    txt_folder = 'txtoutput3/'
    xml_save = 'anno3/'
    xml_ori = 'ann_origin.xml'
    total_trans(txt_folder,xml_save,image_folder,xml_ori)
    
    