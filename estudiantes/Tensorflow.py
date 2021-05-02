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

##leer archivo con pandas
#mat = pd.read_csv('/Users/ianzaidenweber/Desktop/IA/estudiantes/mat_clean.csv')
#mat = pd.read_csv('/Users/bernardoaltamirano/Google Drive/ITAM/9no_Semestre/IA/proyectos/estudiantes-tensorflow/estudiantes/mat_clean.csv')
mat = pd.read_csv('/Users/bernardoaltamirano/Google Drive/ITAM/9no_Semestre/IA/proyectos/estudiantes-tensorflow/estudiantes/student-mat.csv', ';')
mat = mat.drop(['school','sex','address','famsize', 'Pstatus', 'Mjob','Fjob','reason', 'guardian', 'schoolsup', 'activities', 'nursery', 'higher','internet', 'romantic', 'famsup', 'paid'], axis=1)
mat.head()

#Estandarizar datos
mat_norm = (mat - mat.mean()) / mat.std()
mat_norm.head()

#desestandariza para los resultados
y_mean = mat['G3'].mean()
y_std = mat['G3'].std()

def des_estandariza(pred):
  return float(pred * y_std + y_mean)

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
def usar_modelo():
    
    model = Sequential([
        Dense(10, input_shape = (mat.shape[1],), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adam'
    )
    
    return model

model = usar_modelo()
model.summary()

#detiene las iteraciones en caso de que hayan k iteraciones sin mejorar
termina_antes = EarlyStopping(monitor='val_loss', patience = 300)

model = usar_modelo()

#antes de entrenar, hace predicciones aleatorias para comparar con modelo entrenado
preds_on_untrained = model.predict(X_test)

#entrena modelo
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000,
    callbacks = [termina_antes]
)

#perdida de entrenamiento y validacion
plot_loss(history)

#compara predicciones del modelo no-entrenado y entrenado
preds_on_trained = model.predict(X_test)

compare_predictions(preds_on_untrained, preds_on_trained, y_test)

#compara predicciones
score_on_untrained = [des_estandariza(y) for y in preds_on_untrained]
score_on_trained = [des_estandariza(y) for y in preds_on_trained]
score_y_test = [des_estandariza(y) for y in y_test]

compare_predictions(score_on_untrained, score_on_trained, score_y_test)