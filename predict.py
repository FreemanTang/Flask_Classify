# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:18:23 2021

@author: TFX
"""
import tensorflow as tf
import numpy as np
import cv2

class_names = ['bird', 'book', 'butterfly', 'cattle', 'chicken', 'elephant', 'horse', 
               'phone', 'sheep', 'shoes', 'spider', 'squirrel', 'watch']

def pred(img_path,weight_path):
    loaded_model = tf.keras.models.load_model(weight_path)
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img,(32,32))/255.0
    img_dim = np.expand_dims(img_resize,0)
    # print(img.shape)
    pred_arr = loaded_model.predict(img_dim)
    # print(pred_arr)
    # print(np.max(pred_arr[0]))
    class_index = np.argmax(pred_arr[0])
    class_name = class_names[np.argmax(pred_arr[0])]
    # print(np.sum(pred_arr[0]))
    return class_index,class_name

if __name__=="__main__":
    img_path = "imgs/book.jpg"
    class_name = pred(img_path)
    print(class_name)
