# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:40:25 2020

@author: goka
"""

from __future__ import unicode_literals
# import youtube_dl
import os 

# """ 

# This code is to detect a car in a video 

# """

### Now, it's time to write the code to detect the cars in the video frames

# Import librairies of python OpenCV
import cv2 

# capture frames from a video 
cap = cv2.VideoCapture("relaxing-highway-traffic.mp4")

# Training XML classifiers describes some features of some object we want to detect
# Make sure you put this XML file in the same folder as your code
car_cascade = cv2.CascadeClassifier("cars.xml")

# Loop runs if capturing has been initialized 
while True: 
    
    # reads frames from a video 
    ret, frames = cap.read()
    
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    
    # To draw a rectangle in each cars 
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,0,255), 2)
        
    # Display frames in a window 
    cv2.imshow('video2', frames)
    
    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27:
        break 
    
# De allocate any associated memory usage
cv2.destroyAllWindows()

    