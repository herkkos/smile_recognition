# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:56:53 2020

@author: Herkko
"""

'''
Main file for training and testing a neural network which is meant to detect
smiling faces from images in real time.
'''

import cv2
import face_recognition
import glob
import matplotlib.pyplot as plt
import numpy as np
from os import mkdir
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#import tensorflow as tf
from tensorflow.keras import layers, losses, models, optimizers


# Constants for comfort
BATCH_SIZE = 8
DIM = (64, 64)
EPOCHS = 50
FILE = 'GENKI-4K/GENKI-4K_Labels.txt'
FOLDER = 'GENKI-4K/files'
GPU_ACCELERATED = True
NAME = 'te'

faceRecModel = 'cnn' if GPU_ACCELERATED else 'hog'


# Loads jpg images from given folder and returns numpy array containing
# all images.
def loadImages(folder):
    folder = folder + "/*.jpg"
    images = []
    ignored = []
    filenames = [img for img in glob.glob(folder)]
    for i in range(len(filenames)):
        try:
            img = filenames[i]
            image = cv2.imread(img, cv2.IMREAD_COLOR)
            image = image[:, :, ::-1]
            faceLocations = face_recognition.face_locations(image, 1, model=faceRecModel)[0]
            top, right, bottom, left = faceLocations
            foundFace = image[top:bottom, left:right, :]
            resized = cv2.resize(foundFace, DIM, interpolation = cv2.INTER_AREA)
            resized = resized.astype(np.float32)
            resized /= 255.
            images.append(resized)
        except:
            print('Loading failed for {}'.format(img))
            ignored.append(i)
        
    return np.array(images), ignored


# Loads labels for image data from a textfile
def loadLabels(file, ignored):
    labels = []
    f = open(file, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        if i in ignored:
            continue
        line = lines[i]
        # Save only smiling label, ignore Head Pose parameters
        labels.append(int(line.split(' ')[0]))
    return np.array(labels)    
    

# Save model with given name
def saveModel(model, name):
    try:
        mkdir(name)
    except:
        print("Failed to create folder for model")   
    model.save(name)
    
    
#
def saveScore(history, name):
    loss = np.array(history.history['loss'])
    accuracy = np.array(history.history['accuracy'])
    val_loss = np.array(history.history['val_loss'])
    val_accuracy = np.array(history.history['val_accuracy'])
    data = np.stack((loss, accuracy, val_loss, val_accuracy))
    np.savetxt(name + ".csv", data, delimiter=',')


def main():
    # Load data
    X, ignored = loadImages(FOLDER)
    y = loadLabels(FILE, ignored)
    
    # Split data to training and validation(test) data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    # model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    # model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.summary()

    # Define optimizer function
    # opt = optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
    # opt = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.0000001)
    # opt = optimizers.Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.999, epsilon=0.0000001)
    # opt = optimizers.Adadelta(learning_rate=0.1, rho=0.95, epsilon=0.0000001)
    opt = optimizers.Adadelta(learning_rate=0.01, rho=0.95, epsilon=0.00001)

    # Define loss function
    loss = losses.BinaryCrossentropy(label_smoothing=0.2)

    # Compile the model
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    # model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    # Train the model
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot accuracy after each epoch
    plt.plot(history.history['accuracy'])

    # Print score
    y_pred = model.predict_classes(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    
    # Print confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    # Save model
    saveModel(model, NAME)
    saveScore(history, NAME)

main()
