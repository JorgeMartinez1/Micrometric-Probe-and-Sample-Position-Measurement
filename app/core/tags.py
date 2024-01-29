import cv2
import numpy as np
import copy
import core.gestionKalman as gestionKalman
import time
import logging

from core.kalman import GestionKalman

class GestionTags():
    def __init__(self, logger: logging = None, configuracion=(16, 1)):
        super().__init__()
        self.logger = logger
        self.configuracion = configuracion
        self.gestion_kalman = GestionKalman(self.logger)

    def inicializar_tags(self):
        collection_tags = {}

        if self.configuracion == (12, 1):
            collection_tags = inicializar_tags_12_1(12)

        if self.configuracion == (15, 1):
            collection_tags = inicializar_tags_15_1(12, 3, 1)

        if self.configuracion == (16, 1):
            collection_tags = self.inicializar_tags_16_1()

        return collection_tags

    def inicializar_tags_16_1(self):
        nTagsBaseTL = 12
        nTagsBaseBL = 4

        collection_tags = {}
        for n in range(nTagsBaseTL):
            kalman, measurement_array, dt_array = self.gestion_kalman.inicializar_kalman_multiples_puntos(n_puntos=4)
            collection_tags[str(n + 1) + 'TL'] = {'code': None, 'location': 'TL', 'corners': None, 'corners_3D': None,
                                                  'kalman': kalman, 'measurement_array': measurement_array,
                                                  'dt_array': dt_array, 'tiempo_estimacion': time.time(),
                                                  'tipo': 'inicial'}

        for n in range(nTagsBaseBL):
            kalman, measurement_array, dt_array = self.gestion_kalman.inicializar_kalman_multiples_puntos(n_puntos=4)
            collection_tags[str(n + 1) + 'BL'] = {'code': None, 'location': 'BL', 'corners': None, 'corners_3D': None,
                                                  'kalman': kalman, 'measurement_array': measurement_array,
                                                  'dt_array': dt_array, 'tiempo_estimacion': time.time(),
                                                  'tipo': 'inicial'}

        kalman, measurement_array, dt_array = self.gestion_kalman.inicializar_kalman_multiples_puntos(n_puntos=4)
        collection_tags[str(15) + 'BL'] = {'code': None, 'location': 'BL', 'corners': None, 'corners_3D': None,
                                           'kalman': kalman,
                                           'measurement_array': measurement_array, 'dt_array': dt_array,
                                           'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

        return collection_tags