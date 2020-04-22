import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
import random


# path to images
path = './images/animals/'

# animal categories
categories = ['dogs', 'panda', 'cats']
 
# initialize the data
data = []
labels = []
imagePaths = []
HEIGHT = 50
WIDTH = 50
N_CHANNELS = 3

#on cree une liste imagePaths qui contient toutes les images
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

#on melange les images dans notre liste, pour avoir un melange d'images de chien, de chat et de panda
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # on resize les images afin qu'elles aient toutes la meme taille, pui on les stocke dans une liste de data
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    labels.append(label)
    
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")/255
labels = np.array(labels)

# on divise les donnees en quatre groupes : deux groupes de test et deux groupes d'entrainement
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = np_utils.to_categorical(trainY, 3)

# on cree le reseau de neurones
model = Sequential()

model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=32, epochs=25, verbose=1)

# on prend une image dans le set de test et on predit le type d'animal dont il s'agit

predictions = model.predict(testX[:1])
max_chance = 0
animal = 0
for i in range(0,3):
    if (predictions[0][i] > max_chance) :
        max_chance = predictions[0][i]
        animal = i
        
if(animal == 0):
    print("it's a dog")

if(animal == 1):
    print("it's a panda")
    
if(animal == 2):
    print("it's a cat")
    
if(animal == testY[:1]):
    print("Yeeeah ! Success !!")   
else :
    print("Echec")