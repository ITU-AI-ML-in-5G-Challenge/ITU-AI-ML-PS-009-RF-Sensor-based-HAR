import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate,Input,concatenate
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


import math
import matplotlib
import cmath
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height=128
img_width=128
batch_size=32

folder = "images"
classes = 2 # 2-> If the folder contains 2 classes, 3->  If the folder contains 3 classes, 8-> If the folder contains 8 classes

if(classes == 2 or classes == 3):
    filter_size = 48
elif(classes==8):
    filter_size=16

ds_train = keras.preprocessing.image_dataset_from_directory(
    folder,
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    validation_split=0.5,
    subset="training",
    )


ds_validation = keras.preprocessing.image_dataset_from_directory(
    folder,
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    validation_split=0.5,
    subset="validation",
    )

ip = Input(shape = (128, 128, 3))


def naive_inception_module(layer_in, hf, sf):
    
    convH1 = Conv2D(hf, (3,11), padding='same', activation='relu')(layer_in)
    convH2 = Conv2D(hf, (5,21), padding='same', activation='relu')(layer_in)
    conv9 = Conv2D(sf, (9,9), padding='same', activation='relu')(layer_in)
    layer_out = concatenate([convH1, convH2, conv9], axis=-1)

    return layer_out

InputLayer = Input(shape=(img_height, img_width, 3))
IL1 = naive_inception_module(InputLayer, filter_size, filter_size)

MP1 = MaxPool2D((2,2), strides=(2,2), padding='valid')(IL1)
BN1 = BatchNormalization()(MP1)
IL2 = naive_inception_module(BN1, filter_size, filter_size)

MP2 = MaxPool2D((4,4), strides=(4,4), padding='valid')(IL2)
BN2 = BatchNormalization()(MP2)
CL1 = Conv2D(128, (5, 5), activation='relu',padding='valid')(BN2)

MP3 = MaxPool2D((2,2), strides=(2,2), padding='valid')(CL1)
BN3 = BatchNormalization()(MP3)

F = Flatten()(BN3)
D1 = Dense(units=80, activation='relu')(F)
Do1 = Dropout(0.2)(D1)
D2 = Dense(units=80, activation='relu')(Do1)
OL = Dense(units=classes, activation='softmax')(D2)
model = Model(inputs=InputLayer, outputs=OL)

checkpoint_path = "1.h5"
mcp_save = ModelCheckpoint(checkpoint_path, save_best_only=True,save_weights_only=True, monitor='val_accuracy', mode='max')
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
history=model.fit(ds_train, epochs=100, callbacks=[mcp_save], validation_data=ds_validation)
model.evaluate(ds_validation)


folder_test = "images"

ds_test = keras.preprocessing.image_dataset_from_directory(
    folder_test,
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    )

model.load_weights(checkpoint_path)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.evaluate(ds_train,batch_size=batch_size)
model.evaluate(ds_validation,batch_size=batch_size)
model.evaluate(ds_test,batch_size=batch_size)
