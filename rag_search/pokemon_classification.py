import numpy as np 
import pandas as pd 
import os

import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils.image_utils import load_img, img_to_array
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.applications import DenseNet201

from sklearn.metrics import classification_report
import requests

import cv2

directory = r"D:\Program Files\Code repositories\PokemonData"
labels = os.listdir(directory)
nb = len(labels)

print(nb)

stored = {}

def input_target_split(train_dir,labels):
    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(train_dir,label)
        for image in os.listdir(folder):
            
#             print(os.path.join(folder,image))
            try:
                img=load_img(os.path.join(folder,image), target_size=(150,150))
                img=img_to_array(img)
                img=img/255.0
                dataset.append((img,count))
            except:
                pass

        print(f'\rCompleted: {label}',end='')
        stored[label] = count
        count+=1
    random.shuffle(dataset)
    X, y = zip(*dataset)
    
    return np.array(X),np.array(y)

X, y = input_target_split(directory,labels)

print(len(stored))

plt.figure(figsize = (15 , 9))
n = 0
for i in range(15):
    n+=1
    plt.subplot(5 , 5, n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.3)
    plt.imshow(X[i])
    plt.title(f'Label: {labels[y[i]]}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
print(np.unique(y_train,return_counts=True),np.unique(y_test,return_counts=True))

# datagen = ImageDataGenerator(horizontal_flip=True,
#                              vertical_flip=True,
#                              rotation_range=20,
#                              zoom_range=0.2,
#                              width_shift_range = 0.2,
#                              height_shift_range = 0.2,
#                              shear_range=0.1,
#                              fill_mode="nearest")

# testgen = ImageDataGenerator()

# datagen.fit(X_train)
# testgen.fit(X_test)

y_train = np.eye(nb)[y_train]
y_test = np.eye(nb)[y_test]

# img_size = 150
# base_model = DenseNet201(include_top = False,
#                          weights = 'imagenet',
#                          input_shape = (img_size,img_size,3))

# for layer in base_model.layers[:675]:
#     layer.trainable = False

# for layer in base_model.layers[675:]:
#     layer.trainable = True


# model = Sequential()
# model.add(base_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(nb, activation=tf.nn.softmax))
# model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])

# filepath= "model_pokemon.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

# early_stopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
#                                             patience=3, 
#                                             verbose=1, 
#                                             factor=0.2, 
#                                             min_lr=0.00001)

# callbacks_list = [
#         checkpoint,
#         early_stopping,
#         learning_rate_reduction
#     ]

# hist = model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),
#                                         validation_data=testgen.flow(X_test,y_test,batch_size=32),
#                                         epochs=50,
#                                         callbacks=callbacks_list)

model_path = r"D:\Program Files\Code repositories\RAG\RAG\model_pokemon.h5"
model = load_model(model_path)

y_pred = model.predict(X_test)
pred = np.argmax(y_pred,axis=1)
print(pred)

ground = np.argmax(y_test,axis=1)

y_pred = np.argmax(y_pred,axis=1)

y_true = np.argmax(y_test,axis=1)

plt.figure(figsize = (15 , 9))
n = 0
for i in range(len(X_test)):
    if y_pred[i] != y_true[i]:
        n+=1
        if n <= 25:
            plt.subplot(5 , 5, n)
            plt.subplots_adjust(hspace = 0.8 , wspace = 0.3)
            plt.imshow(X_test[i])
            plt.title(f'Actual: {labels[y_true[i]]}\nPredicted: {labels[y_pred[i]]}')

# image = cv2.imread('D:\Program Files\Code repositories\PokemonData\Snorlax\0ba65a40b17147a394dc3b860dc95c46.jpg')
# img = cv2.resize(image, (150, 150))
# img=img/255.0
# img = np.expand_dims(img, axis=0)
# pred = model.predict(img)
# label = np.argmax(pred,axis=1)
# print(labels[label[0]])

# pokemon = labels[label[0]].lower()
# url = f'https://pokeapi.co/api/v2/pokemon/{pokemon}'
# r = requests.get(url)

# print("Name: ",r.json()['name'])
# print("Base Experience: ",r.json()['base_experience'])
# print("Height: ",r.json()['height'],'m')
# print("Weight: ",r.json()['weight'],'kg')

try:
    image_path = r'D:\Program Files\Code repositories\PokemonData\Snorlax\test.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Cannot open or read the image file at path: {image_path}")
    
    img = cv2.resize(image, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = np.argmax(pred, axis=1)
    print(labels[label[0]])

    pokemon = labels[label[0]].lower()
    url = f'https://pokeapi.co/api/v2/pokemon/{pokemon}'
    r = requests.get(url)

    print("Name: ", r.json()['name'])
    print("Base Experience: ", r.json()['base_experience'])
    print("Height: ", r.json()['height'], 'm')
    print("Weight: ", r.json()['weight'], 'kg')

except FileNotFoundError as e:
    print(e)
except cv2.error as e:
    print(f"OpenCV error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")