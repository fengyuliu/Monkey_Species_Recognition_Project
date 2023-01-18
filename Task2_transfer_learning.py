from tensorflow.keras.applications import VGG19
from tensorflow.keras import applications
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from pathlib import Path
from io import BytesIO
from PIL import Image
from sklearn import metrics

import random
import cv2
import math
import scipy
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os


# information of the images
height = 224
width = 224
channels = 3
num_class = 10
batch_size = 32
epochs = 10


# location of the sets
train_set = './archive/training/training/'
validation_set = './archive/validation/validation/'
train_name = os.listdir(train_set)
validation_name = os.listdir(validation_set)

D1=[]
D2=[]
for folder in train_name:
    image_list=os.listdir(train_set+"/"+folder)
    for N in image_list:
        img=image.load_img(train_set+"/"+folder+"/"+N,target_size=(height,width))
        img=image.img_to_array(img)
        img=preprocess_input(img)
        D1.append(img)
        D2.append(train_name.index(folder))

validation_imag=[]
validation_Ori=[]
validation_label=[]
for F in validation_name:
    image_list=os.listdir(validation_set+"/"+F)
    for N in image_list:
        img=image.load_img(validation_set+"/"+F+"/"+N,target_size=(height,width))
        img=image.img_to_array(img)
        validation_Ori.append(img.copy())
        img=preprocess_input(img)
        validation_imag.append(img)
        validation_label.append(validation_name.index(F))

D1=np.array(D1)
D2=to_categorical(D2) 
Dtrain, Dtest, Ltrain, Ltest = train_test_split(D1,D2,test_size=0.2,random_state=5)
validation_imag=np.array(validation_imag) 
validation_label=to_categorical(validation_label)

# build the model
inputs=layers.Input(shape=(height,width,channels))
model_vgg = VGG19(
    weights='imagenet',
    input_tensor=inputs,
    include_top=False
    )  
MM = model_vgg.output
MM = layers.Flatten()(MM)
outputs = layers.Dense(num_class,activation='softmax')(MM)
model = models.Model(inputs=inputs,outputs=outputs)

# change the number to freeze and un freeze layers. Set it to -1 when only the last layer need to be trained
for layer in model.layers[:-4]:
    layer.trainable=False

# choose optimizer=optimizers.SGD(lr=1e-4) or 'adam'
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.SGD(lr=1e-3),
    metrics=['accuracy'])
model.summary()


# training
history = model.fit(
    Dtrain,
    Ltrain,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(Dtest,Ltest),
    verbose=True
)


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8,5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()

plot_learning_curves(history, 'accuracy', epochs, 0, 1.1)
plot_learning_curves(history, 'loss', epochs, 0, 5)

