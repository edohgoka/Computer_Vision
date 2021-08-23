# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:21:47 2020

@author: goka
"""

""" 
Python Face Detection 

"""
import cv2 

# load the cascade 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the input image 
img = cv2.imread("workplace-1245776_1920.jpg")

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces 
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces 
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
