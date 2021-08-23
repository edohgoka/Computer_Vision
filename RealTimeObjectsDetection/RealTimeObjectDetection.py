# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:45:14 2021

@author: goken
"""

from imageai.Detection import ObjectDetection
import os 

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

## Google to download this resnet50_coco_best_v2.1.0.h5 file 
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))

detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 
                                                                      "laptop.jpg"), 
                                             output_image_path=os.path.join(execution_path,
                                                                       "imageNew_5.jpg"))

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])