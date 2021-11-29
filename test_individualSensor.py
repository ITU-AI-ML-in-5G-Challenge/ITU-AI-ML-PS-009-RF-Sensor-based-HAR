#Hierarchical classifier for ITU-PS-009
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate,Input,concatenate
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, UpSampling2D,MaxPool2D
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import math
import numpy as np
import matplotlib
import cmath
import matplotlib.pyplot as plt
import cv2

import os 
import glob 


img_height=128
img_width=128

sensor='Xethru' #This can be set to 77GHz,24GHz,Xethru

inp_path_sel = 1 # 0 -> Path to single image, 1-> path to a folder containing images

path = "images_Xethru/8/" # provide path to image/folder based on inp_path_sel

def func(string):
    if(string=='77GHz'):
        return 0
    elif (string=='24GHz'):
        return 1
    elif (string=='Xethru'):
        return 2


img_height=128
img_width=128
dataset = func(sensor) #Initializing the dataset variable based on the sensor.
input_img=Input(shape=(img_height,img_width,3))

def naive_inception_module(layer_in, hf, sf):
    convH1 = Conv2D(hf, (3,11), padding='same', activation='relu')(layer_in)
    convH2 = Conv2D(hf, (5,21), padding='same', activation='relu')(layer_in)
    conv9 = Conv2D(sf, (9,9), padding='same', activation='relu')(layer_in)
    layer_out = concatenate([convH1, convH2, conv9], axis=-1)
    return layer_out

InputLayer = Input(shape=(img_height, img_width, 3))
InLayerOP = naive_inception_module(InputLayer, 48, 48)

MP1 = MaxPool2D((2,2), strides=(2,2), padding='valid')(InLayerOP)
BN1 = BatchNormalization()(MP1)
IL2 = naive_inception_module(BN1,  48, 48)
MP2 = MaxPool2D((4,4), strides=(4,4), padding='valid')(IL2)
BN2 = BatchNormalization()(MP2)
CL4 = Conv2D(128, (5, 5), activation='relu',  padding='valid')(BN2)
MP3 = MaxPool2D((2,2), strides=(2,2), padding='valid')(CL4)
BN3 = BatchNormalization()(MP3)

F = Flatten()(BN3)
D1 = Dense(units=80, activation='relu')(F)
Do1 = Dropout(0.2)(D1)
D2 = Dense(units=80, activation='relu')(Do1)
Do2 = Dropout(0.2)(D2)
OL = Dense(units=2, activation='softmax')(D2)
model_1 = Model(inputs=InputLayer, outputs=OL)

InputLayer = Input(shape=(img_height, img_width, 3))
InLayerOP = naive_inception_module(InputLayer,  48, 48)

MP1 = MaxPool2D((2,2), strides=(2,2), padding='valid')(InLayerOP)
BN1 = BatchNormalization()(MP1)
IL2 = naive_inception_module(BN1, 48, 48)
MP2 = MaxPool2D((4,4), strides=(4,4), padding='valid')(IL2)
BN2 = BatchNormalization()(MP2)
CL4 = Conv2D(128, (5, 5), activation='relu', padding='valid')(BN2)
MP3 = MaxPool2D((2,2), strides=(2,2), padding='valid')(CL4)
BN3 = BatchNormalization()(MP3)

F = Flatten()(BN3)
D1 = Dense(units=80, activation='relu')(F)
Do1 = Dropout(0.2)(D1)
D2 = Dense(units=80, activation='relu')(Do1)
Do2 = Dropout(0.2)(D2)
OL = Dense(units=3, activation='softmax')(D2)
model_2 = Model(inputs=InputLayer, outputs=OL)

InputLayer = Input(shape=(img_height, img_width, 3))
InLayerOP = naive_inception_module(InputLayer, 16, 16)

MP1 = MaxPool2D((2,2), strides=(2,2), padding='valid')(InLayerOP)
BN1 = BatchNormalization()(MP1)
IL2 = naive_inception_module(BN1, 16, 16)
MP2 = MaxPool2D((4,4), strides=(4,4), padding='valid')(IL2)
BN2 = BatchNormalization()(MP2)
CL4 = Conv2D(128, (5, 5), activation='relu', padding='valid')(BN2)
MP3 = MaxPool2D((2,2), strides=(2,2), padding='valid')(CL4)
BN3 = BatchNormalization()(MP3)

F = Flatten()(BN3)
D1 = Dense(units=80, activation='relu')(F)
Do1 = Dropout(0.2)(D1)
D2 = Dense(units=80, activation='relu')(Do1)
Do2 = Dropout(0.2)(D2)
OL = Dense(units=8, activation='softmax')(D2)
model_3 = Model(inputs=InputLayer, outputs=OL)



if(dataset==0):
    
    #Checkpoint path to load the saved models
    checkpoint_path =  "77_8_9_10_classes_inception.h5"
    checkpoint_path_1 = "77_2_3_classes_inception.h5"
    checkpoint_path_2 = "77_remaining_classes_inception.h5"

    model_2.load_weights(checkpoint_path)
    model_2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model_1.load_weights(checkpoint_path_1)
    model_1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model_3.load_weights(checkpoint_path_2)
    model_3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


elif(dataset==1):
    #Checkpoint path to load the saved models
    checkpoint_path =  "24_8_9_10_classes_inception.h5"
    checkpoint_path_1 = "24_2_3_classes_inception.h5"
    checkpoint_path_2 = "24_remaining_classes_inception.h5"
    
    model_2.load_weights(checkpoint_path)
    model_2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model_1.load_weights(checkpoint_path_1)
    model_1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model_3.load_weights(checkpoint_path_2)
    model_3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


elif(dataset==2):
    #Checkpoint path to load the saved models
    checkpoint_path =  "Xethru_8_9_10_classes_inception.h5"
    checkpoint_path_1 = "Xethru_2_3_classes_inception.h5"
    checkpoint_path_2 = "Xethru_remaining_classes_inception.h5"

    model_2.load_weights(checkpoint_path)
    model_2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model_1.load_weights(checkpoint_path_1)
    model_1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model_3.load_weights(checkpoint_path_2)
    model_3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])




if(dataset==0):
    
    
    #Hierarchical N/W 
    def ensemble(ip):
        y=model_3(ip)
        a=tf.argmax(y,axis=1)
        b=tf.cast(0,dtype=tf.int64)
        c=tf.cast(1,dtype=tf.int64)
        d=tf.cast(2,dtype=tf.int64)
        e=tf.cast(7,dtype=tf.int64)
        f=tf.cast(3,dtype=tf.int64)
        g=tf.cast(4,dtype=tf.int64)
        h=tf.cast(5,dtype=tf.int64)
        i=tf.cast(6,dtype=tf.int64)

        if(tf.reduce_all(tf.equal(a, d))):
            y_1=model_1(ip)
            a_1=tf.argmax(y_1,axis=1)
            if(tf.reduce_all(tf.equal(a_1, b))):
                return 2
            elif(tf.reduce_all(tf.equal(a_1, c))):
                 return 3
        elif(tf.reduce_all(tf.equal(a, e))):
             y_1=model_2(ip)
             a_1=tf.argmax(y_1,axis=1)
             if(tf.reduce_all(tf.equal(a_1, b))) :
                return 8
             elif(tf.reduce_all(tf.equal(a_1, c))):
                 return 9
             elif(tf.reduce_all(tf.equal(a_1, d))):
                 return 10
        else:
            if(tf.reduce_all(tf.equal(a, b))):
                 return 0
            elif(tf.reduce_all(tf.equal(a, c))):
                 return 1
            elif(tf.reduce_all(tf.equal(a, f))):
                 return 4
            elif(tf.reduce_all(tf.equal(a, g))):
                 return 5
            elif(tf.reduce_all(tf.equal(a, h))):
                 return 6
            elif(tf.reduce_all(tf.equal(a, i))):
                 return 7


    if inp_path_sel == 0:

        img = cv2.imread(path)
        dim=(img_height,img_width)
        img=cv2.resize(img,dim)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.reshape(img_rgb,(1,img_height,img_width,3))
        
        # Predicting the output
        y_pred = ensemble(tf.cast(img_rgb, tf.float32))
        print(y_pred)


    elif inp_path_sel == 1:

        data_path = os.path.join(path,'*g') 

        files = glob.glob(data_path)


        for f1 in files: 
            img = cv2.imread(f1)
            dim=(img_height,img_width)
            img=cv2.resize(img,dim)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.reshape(img_rgb,(1,img_height,img_width,3))
            
            # Predicting the output
            y_pred = ensemble(tf.cast(img_rgb, tf.float32))
            print(y_pred)

elif(dataset==1):
    

    #Hierarchical N/W 
    def ensemble(ip):
        y=model_3(ip)
        a=tf.argmax(y,axis=1)
        b=tf.cast(0,dtype=tf.int64)
        c=tf.cast(1,dtype=tf.int64)
        d=tf.cast(2,dtype=tf.int64)
        e=tf.cast(7,dtype=tf.int64)
        f=tf.cast(3,dtype=tf.int64)
        g=tf.cast(4,dtype=tf.int64)
        h=tf.cast(5,dtype=tf.int64)
        i=tf.cast(6,dtype=tf.int64)
        

        if(tf.reduce_all(tf.equal(a, d))):
            y_1=model_1(ip)
            a_1=tf.argmax(y_1,axis=1)
            if(tf.reduce_all(tf.equal(a_1, b))):
                return 2
            elif(tf.reduce_all(tf.equal(a_1, c))):
                 return 3
        elif(tf.reduce_all(tf.equal(a, e))):
             y_1=model_2(ip)
             a_1=tf.argmax(y_1,axis=1)
             if(tf.reduce_all(tf.equal(a_1, b))) :
                return 8
             elif(tf.reduce_all(tf.equal(a_1, c))):
                 return 9
             elif(tf.reduce_all(tf.equal(a_1, d))):
                 return 10
        else:
            if(tf.reduce_all(tf.equal(a, b))):
                 return 0
            elif(tf.reduce_all(tf.equal(a, c))):
                 return 1
            elif(tf.reduce_all(tf.equal(a, f))):
                 return 4
            elif(tf.reduce_all(tf.equal(a, g))):
                 return 5
            elif(tf.reduce_all(tf.equal(a, h))):
                 return 6
            elif(tf.reduce_all(tf.equal(a, i))):
                 return 7


    
    if inp_path_sel == 0:

        img = cv2.imread(path)
        dim=(img_height,img_width)
        img=cv2.resize(img,dim)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.reshape(img_rgb,(1,img_height,img_width,3))
        
        # Predicting the output
        y_pred = ensemble(tf.cast(img_rgb, tf.float32))
        print(y_pred)


    elif inp_path_sel == 1:

        data_path = os.path.join(path,'*g') 

        files = glob.glob(data_path)


        for f1 in files: 
            img = cv2.imread(f1)
            dim=(img_height,img_width)
            img=cv2.resize(img,dim)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.reshape(img_rgb,(1,img_height,img_width,3))
            # Predicting the output
            y_pred = ensemble(tf.cast(img_rgb, tf.float32))
            print(y_pred)
       
        
    
elif(dataset==2):
    
    #Hierarchical N/W 
    def ensemble(ip):
        y=model_3(ip)
        a=tf.argmax(y,axis=1)
        b=tf.cast(0,dtype=tf.int64)
        c=tf.cast(1,dtype=tf.int64)
        d=tf.cast(2,dtype=tf.int64)
        e=tf.cast(7,dtype=tf.int64)
        f=tf.cast(3,dtype=tf.int64)
        g=tf.cast(4,dtype=tf.int64)
        h=tf.cast(5,dtype=tf.int64)
        i=tf.cast(6,dtype=tf.int64)


        if(tf.reduce_all(tf.equal(a, d))):
            y_1=model_1(ip)
            a_1=tf.argmax(y_1,axis=1)
            if(tf.reduce_all(tf.equal(a_1, b))):
                return 2
            elif(tf.reduce_all(tf.equal(a_1, c))):
                 return 3
        elif(tf.reduce_all(tf.equal(a, e))):
             y_1=model_2(ip)
             a_1=tf.argmax(y_1,axis=1)
             if(tf.reduce_all(tf.equal(a_1, b))) :
                return 8
             elif(tf.reduce_all(tf.equal(a_1, c))):
                 return 9
             elif(tf.reduce_all(tf.equal(a_1, d))):
                 return 10
        else:
            if(tf.reduce_all(tf.equal(a, b))):
                 return 0
            elif(tf.reduce_all(tf.equal(a, c))):
                 return 1
            elif(tf.reduce_all(tf.equal(a, f))):
                 return 4
            elif(tf.reduce_all(tf.equal(a, g))):
                 return 5
            elif(tf.reduce_all(tf.equal(a, h))):
                 return 6
            elif(tf.reduce_all(tf.equal(a, i))):
                 return 7


    if inp_path_sel == 0:

        img = cv2.imread(path)
        dim=(img_height,img_width)
        img=cv2.resize(img,dim)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.reshape(img_rgb,(1,img_height,img_width,3))
        
        # Predicting the output
        y_pred = ensemble(tf.cast(img_rgb, tf.float32))
        print(y_pred)


    elif inp_path_sel == 1:

        data_path = os.path.join(path,'*g') 

        files = glob.glob(data_path)


        for f1 in files: 
            img = cv2.imread(f1)
            dim=(img_height,img_width)
            img=cv2.resize(img,dim)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.reshape(img_rgb,(1,img_height,img_width,3))
            
            # Predicting the output
            y_pred = ensemble(tf.cast(img_rgb, tf.float32))
            print(y_pred)






