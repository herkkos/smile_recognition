# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:02:39 2020

@author: Herkko
"""

'''
File for using trained neural network which detects smiles
'''


import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model
import time


DIM = (64, 64)
GPU_ACCELERATED = True
NAME = 'model'

faceRecModel = 'cnn' if GPU_ACCELERATED else 'hog'


def loadModel(name):
    model = load_model(name)
    return model


def main():
    model = loadModel(NAME)
    videoStream = cv2.VideoCapture("testivideo.mp4")
    while(True):
        tik = time.time()
        
        # Read frame from video stream
        ret, frame = videoStream.read()
        
        # Switch colorscheme
        frame = frame[:, :, ::-1]
        
        # Find possible faces
        face_locations = face_recognition.face_locations(frame, model=faceRecModel)
        
        for face in face_locations:
            # Determine smile or not
            top, right, bottom, left = face
            foundFace = frame[top:bottom, left:right, :]
            resized = cv2.resize(foundFace, DIM, interpolation = cv2.INTER_AREA)
            resized = np.expand_dims(resized, axis=0)
            prediction = model.predict_classes(resized)[0,0]
            
            if (prediction):
                label = 'smile'
                cvframe = cv2.UMat(frame)
                cv2.rectangle(cvframe, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(cvframe, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 200), 2)
                frame = cv2.UMat.get(cvframe)
            else:
                label = 'no smile'
                cvframe = cv2.UMat(frame)
                cv2.rectangle(cvframe, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(cvframe, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 200), 2)
                frame = cv2.UMat.get(cvframe)
        
        # Invert colors back
        frame = frame[:, :, ::-1]
        
        # Calculate FPS and show frame
        tok = time.time();
        fps = str(1 / (tok - tik + 0.001))
        cv2.imshow("Webcam stream", frame)
        cv2.setWindowTitle("Webcam stream", "Frames per second: {}".format(fps))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoStream.release()
    cv2.destroyAllWindows()


main()

