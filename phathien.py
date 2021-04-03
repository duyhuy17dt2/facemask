import cv2
import numpy as np
import argparse
import csv
import os
# -*- coding: utf-8 -*-
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def TakeImages():
    name = input("nhap Ten thu muc:\n")
    # Path
    cam = cv2.VideoCapture(0)
    sampleNum = 0
    takephoto = 0
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    cap = cv2.VideoCapture(0)
    sampleNum=0
    takephoto=0
    while(True):
        _, frame = cap.read()
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections =net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                startX = startX - 20
                endX = endX + 20
                startY = startY - 50
                endY = endY+20
                    #incrementing sample number
                if takephoto==1:
                    sampleNum=sampleNum+1
                        #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("E:/Face-Mask-Detection-master/dataset/" + name + "/" + str(sampleNum) + ".jpg", frame[startY:endY,startX:endX])
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,str(sampleNum),(50,50), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,'Nhan nut C de chup anh',(200,50), font, 0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
            #wait for 100 miliseconds
        if cv2.waitKey(10) & 0xFF == ord('c'):
            if takephoto==0:
                takephoto = 1
            else:
                takephoto = 0
            # break if the sample number is morethan 100
        if sampleNum>999:
            break
    cap.release()
    cv2.destroyAllWindows()
TakeImages()