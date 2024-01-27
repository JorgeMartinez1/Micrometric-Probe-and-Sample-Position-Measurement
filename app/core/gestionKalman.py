#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase para el manejo de los tags de Realidad Aumentada

Created on Thu Aug 11 16:13:53 2022
@author: jolumartinez
"""

import cv2
import numpy as np
import copy


def prediccion_kalman_con_correccion(kalman, measurement_array, puntos, dt_array, variacion_tiempo):
    pred = []
    input_points = np.float32(np.ndarray.flatten(puntos))
    for i, j in zip(measurement_array, dt_array):
        kalman.transitionMatrix[i, j] = variacion_tiempo
    # Prediction step
    tp = kalman.predict()

    for i in measurement_array:
        pred.append(tp[i])

    # Correction Step
    kalman.correct(input_points)

    return pred


def prediccion_kalman_sin_correccion(kalman, measurement_array, dt_array, variacion_tiempo):
    pred = []

    for i, j in zip(measurement_array, dt_array):
        kalman.transitionMatrix[i, j] = variacion_tiempo

    tp = kalman.predict()

    for i in measurement_array:
        pred.append(tp[i])

    return pred


def inicializar_kalman_microscopio(configuracion=(12 , 1)):
    esquinasTags = 4;

    nPuntosBase = esquinasTags*configuracion[0]
    nPuntosPunta = esquinasTags*configuracion[1]

    kalmanBase, measurement_array_base, dt_array_base = inicializar_kalman_multiples_puntos(nPuntosBase)
    kalmanPunta, measurement_array_punta, dt_array_punta = inicializar_kalman_multiples_puntos(nPuntosPunta)

    return (kalmanBase, measurement_array_base, dt_array_base),(kalmanPunta, measurement_array_punta, dt_array_punta)


def inicializar_kalman_multiples_puntos(n_puntos):
    n_states = n_puntos * 4
    n_measures = n_puntos * 2

    kalman = cv2.KalmanFilter(n_states, n_measures)
    kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
    kalman.processNoiseCov = np.eye(n_states, dtype=np.float32)*0.001
    kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 0.01

    kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)
    dt = 0.1

    measurement_array = []
    dt_array = []

    for i in range(0, n_states, 4):
        measurement_array.append(i)
        measurement_array.append(i + 1)

    # print(measurement_array)

    for i in range(0, n_states):
        if i not in measurement_array:
            dt_array.append(i)

    # Transition Matrix for [x,y,x',y'] for n such points
    # format of first row [1 0 dt 0 .....]
    for i, j in zip(measurement_array, dt_array):
        kalman.transitionMatrix[i, j] = dt

    # Measurement Matrix for [x,y,x',y'] for n such points
    # format of first row [1 0 0 0 .....]
    for i in range(0, n_measures):
        kalman.measurementMatrix[i, measurement_array[i]] = 1

    return kalman, measurement_array, dt_array


def inicializar_kalman_multiples_puntos_3D(n_puntos=1):
    n_states = n_puntos * 6
    n_measures = n_puntos * 3

    kalman = cv2.KalmanFilter(n_states, n_measures)
    kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
    kalman.processNoiseCov = np.eye(n_states, dtype=np.float32)*0.001
    kalman.measurementNoiseCov = np.eye(n_measures, dtype=np.float32) * 0.01

    kalman.measurementMatrix = np.zeros((n_measures, n_states), np.float32)
    dt = 0.1

    measurement_array = []
    dt_array = []

    for i in range(0, n_states, 4):
        measurement_array.append(i)
        measurement_array.append(i + 1)

    # print(measurement_array)

    for i in range(0, n_states):
        if i not in measurement_array:
            dt_array.append(i)

    # Transition Matrix for [x,y,x',y'] for n such points
    # format of first row [1 0 dt 0 .....]
    for i, j in zip(measurement_array, dt_array):
        kalman.transitionMatrix[i, j] = dt

    # Measurement Matrix for [x,y,x',y'] for n such points
    # format of first row [1 0 0 0 .....]
    for i in range(0, n_measures):
        kalman.measurementMatrix[i, measurement_array[i]] = 1

    return kalman, measurement_array, dt_array
