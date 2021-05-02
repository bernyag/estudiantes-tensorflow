#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 08:57:47 2021

@author: ianzaidenweber
"""


import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

#%matplotlib inline
#tf.logging.set_verbosity(tf.logging.ERROR)

##leer archivo con pandas
mat = pd.read_csv('/Users/ianzaidenweber/Desktop/IA/estudiantes/mat_clean.csv')
mat.head()

#Estandarizar datos
mat_norm = (mat - mat.mean()) / mat.std()
mat_norm.head()

#desestandariza para los resultados
y_mean = mat['G3'].mean()
y_std = mat['G3'].std()

def convert_label_value(pred):
  return int(pred * y_std + y_mean)

print(convert_label_value(0.350088))


#crear columna con diferencia de parcial 2 y parcial 1
mat_norm['Grade_dif']=mat_norm.G2-mat_norm.G1

#crear df para variable objetivo y para variables de soporte
Y = mat_norm.G3 
X = mat_norm.drop('G3', axis=1)

#hacer el split de test y train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


#extraer valores numericos unicamente
X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape)
print('Y_arr shape: ', Y_arr.shape)

#
def get_model():
    
    model = Sequential([
        Dense(10, input_shape = (28,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adadelta'
    )
    
    return model

model = get_model()
model.summary()

#detiene las iteraciones en caso de que hayan k iteraciones sin mejorar
early_stopping = EarlyStopping(monitor='val_loss', patience = 5)

model = get_model()

#antes de entrenar, hace predicciones aleatorias para comparar con modelo entrenado
preds_on_untrained = model.predict(X_test)

#entrena modelo
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000,
    callbacks = [early_stopping]
)

#perdida de entrenamiento y validacion
plot_loss(history)

#compara predicciones del modelo no-entrenado y entrenado
preds_on_trained = model.predict(X_test)

compare_predictions(preds_on_untrained, preds_on_trained, y_test)

#compara predicciones
price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_on_untrained, price_on_trained, price_y_test)



