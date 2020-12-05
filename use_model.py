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
from tensorflow.keras.models import load_model

GPU_ACCELERATED = True
NAME = 'model'

faceRecModel = 'cnn' if GPU_ACCELERATED else 'hog'


def loadModel(name):
    model = load_model(name)
    return model


def main():
    model = loadModel(NAME)
    videoStream = cv2.VideoCapture(0)
    while(True):
        # Read frame from video stream
        ret, frame = videoStream.read()
        
        # Switch colorscheme
        frame = frame[:, :, ::-1]
        
        # Find possible faces
        face_locations = face_recognition.face_locations(frame, 1, model=faceRecModel)
        
        for face in face_locations:
            # Determine smile or not
            top, right, bottom, left = face
            foundFace = frame[top:bottom, left:right, :]
            prediction = round(model.predict(foundFace)[0,0])
            if (prediction):
                label = 'smile'
                cv2.putText(frame, label)
            else:
                label = 'no smile'
                cv2.putText(frame, label)
        
        # Run smile recognition on the frame
        prediction = model.predict(frame)
            
        cv2.imshow('Stream', frame)
        
        if cv2.waitKey(0):
            break

    videoStream.release()
    cv2.destroyAllWindows()


main()

