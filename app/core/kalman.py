import logging
import cv2
import numpy as np
import copy
class GestionKalman():
    def __init__(self, logger: logging = None):
        super().__init__()
        self.logger = logger
        self.logger.info("Se inicializa la clase para Kalman")

    def inicializar_kalman_multiples_puntos(self, n_puntos):
        n_states = n_puntos * 4
        n_measures = n_puntos * 2

        kalman = cv2.KalmanFilter(n_states, n_measures)
        kalman.transitionMatrix = np.eye(n_states, dtype=np.float32)
        kalman.processNoiseCov = np.eye(n_states, dtype=np.float32) * 0.001
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