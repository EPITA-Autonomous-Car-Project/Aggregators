import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.image as npimg
from squeezenet import SqueezeNet, SqueezeNet_11

import pandas as pd

datadir = 'Self-Driving-Car'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(('driving_log.csv'), names = columns)

pd.set_option('display.max_colwidth', -1)
data.head()

dataframe = data[['center','steering','throttle','reverse','speed']]

dataframe['center1'] = dataframe['center'].str[31:]

dataframe.head()

def load_img_steering(df):
    image_path = []
    steering = []
    for i in range(len(dataframe)):
        indexed_data = dataframe.iloc[i]
        center= indexed_data[5]
        image_path.append(os.path.join('C:\\Users\\admin\\Documents\\auto lab1\\self-driving-car-master', center.strip()))
        steering.append((indexed_data[1].astype(int)))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering( dataframe)
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=0)

def img_preprocess(img):
    
    img = npimg.imread(img)
    ## Crop image to remove unnecessary features
    img = img[60:135, :, :]
    ## Change to YUV image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    ## Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ## Decrease size for easier processing
    img = cv2.resize(img, (100, 100))
    ## Normalize values
    img = img / 255
    return img
X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))
(unique, counts) = np.unique(dataframe["steering"],return_counts=True)
len(unique)

utils = tf.keras.utils
y_train = utils.to_categorical(Y_train, num_classes=302)
y_test = utils.to_categorical(Y_valid, num_classes=302)
print(y_train.shape)
print(y_test.shape)

model_11 = SqueezeNet_11(input_shape=(100,100,3), nb_classes=302)
model_11.summary()

losses = tf.keras.losses
optimizers = tf.keras.optimizers 
metrics = tf.keras.metrics
def compile_model(model):

    # loss
    loss = losses.categorical_crossentropy

    # optimizer
    optimizer = optimizers.RMSprop(lr=0.0001)

    # metrics
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]

    # compile model with loss, optimizer, and evaluation metrics
    model.compile(optimizer, loss, metric)

    return model

model_11 = compile_model(model_11)

history = model_11.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_test), batch_size=32, verbose = 1, shuffle=1)

def plot_accuracy_and_loss(history):
    plt.figure(1, figsize= (15, 10))

    # plot train and test accuracy
    plt.subplot(221)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('SqueezeNet accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot train and test loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('SqueezeNet loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.show()

plot_accuracy_and_loss(history)
