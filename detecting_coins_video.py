#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:28:59 2017

@author: julien
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model=load_model('coin_models.hdf5')
model.load_weights('coins_weights.h5')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    img = frame[:,:,0]    
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=40,maxRadius=60)
      
    if circles is not None:
#        print circles
        for i in circles[0,:]:
            # draw the outer circle
            frame = frame.copy()
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            x = model.predict_classes(np.array(i[2]).reshape(1,1),verbose=0)[0]
            if x == 0:
                value = 'R0.5'
            elif x == 1:
                value = 'R2'
            else:
                value = 'R5'
            cv2.putText(frame,value,(i[0],i[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    else:
        print 'out'
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
       
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=40,maxRadius=60)


for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv2.putText(cimg,str(i[2]),(i[0],i[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)


cv2.namedWindow('detected circles',cv2.WINDOW_NORMAL)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()