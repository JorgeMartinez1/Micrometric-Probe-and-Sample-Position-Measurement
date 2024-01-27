#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase para el control de microscopio de microondas con estimación de distancias micrometricas en escenas 3D.

Created on Thu Jan 17 18:27:53 2024
@author: jolumartinez
"""

import cv2
import os.path
import logging
import pyvisa
import serial
import math
from scipy.interpolate import interp1d
import time
import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn-pastel')
import matplotlib.image as mpimg
import imutils
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from PIL.ExifTags import GPSTAGS
from datetime import datetime
import sys
import tkinter as tk
from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
import threading
import random
import core.structure as structure
import core.processor as processor
import copy
from queue import Queue

import core.gestionKalman as gestionKalman
import core.gestionTags as gestionTags
from core.tags import GestionTags
from core.video_devices import VideoDevices

from reportlab.lib.pagesizes import mm
from reportlab.pdfgen import canvas

class MicroEscenesReconstruction():
    def __init__(self, logger: logging = None, caps = None, configuracion:tuple = (16, 1), path_params: str = None,
                 cam_calib: bool = False, qtdp: Queue = None):
        super().__init__()
        self.logger = logger
        self.caps = caps
        self.cap_left = self.caps.cap_left
        self.cap_right = self.caps.cap_right
        self.configuracion = configuracion
        self.outputs_tdparams_folder_path = path_params
        self.cam_calib = cam_calib
        self.queue_td_points = qtdp

        self.gestion_tags = GestionTags(self.logger)

        self.xs = []
        self.ys = []
        self.zs = []
        self.tiempos = []
        self.prom_xs = []
        self.prom_ys = []
        self.prom_zs = []

        self.patron_tags_12 = cv2.imread(f"{self.outputs_tdparams_folder_path}{os.sep}tableroBase12Tags.jpg")  # Patrón sin bordes para pintar en imagen 3d
        self.tag_punta = cv2.imread(f"{self.outputs_tdparams_folder_path}{os.sep}tagPunta.png")  # Patrón con el tag de la punta 15BL

        self.K1 = None
        self.D1 = None
        self.K2 = None
        self.D2 = None

        if self.cam_calib is True:
            self.KL = np.load(f"{self.outputs_tdparams_folder_path}{os.sep}kl.npy")
            self.DL = np.load(f"{self.outputs_tdparams_folder_path}{os.sep}dl.npy")
            self.KR = np.load(f"{self.outputs_tdparams_folder_path}{os.sep}kr.npy")
            self.DR = np.load(f"{self.outputs_tdparams_folder_path}{os.sep}dr.npy")


        # Se inicializan los diccionarios de tags
        self.tags_collection_izq = gestionTags.inicializar_tags(configuracion=self.configuracion)
        self.tags_collection_der = gestionTags.inicializar_tags(configuracion=self.configuracion)
        self.tags_collection_patron = gestionTags.inicializar_tags(configuracion=self.configuracion)

        # Se inicializa el diccionario de tags sintéticos
        self.all_synthetic_collection_izq = gestionTags.inicializar_tags(configuracion=self.configuracion)
        self.all_synthetic_collection_der = gestionTags.inicializar_tags(configuracion=self.configuracion)

        # tags_collection_patron = gestionTags.set_tags_patron_15_1_manual(tags_collection_patron)
        self.tags_collection_patron = gestionTags.set_tags_patron_16_1_manual(self.tags_collection_patron)

        # Se inicializa el objeto kalman para la punta
        self.kalman_punta, self.measurement_array_punta, self.dt_array_punta = gestionKalman.inicializar_kalman_multiples_puntos_3D(
            n_puntos=1)
        self.tiempo_estimacion_punta = time.time()

        self.kalman_proyeccion_punta_fondo, self.measurement_array_proyeccion_punta_fondo, self.dt_array_proyeccion_punta_fondo = gestionKalman.inicializar_kalman_multiples_puntos(
            n_puntos=1)
        self.tiempo_estimacion_proyeccion_punta_fondo = time.time()

        self.PL = np.load(f"{self.outputs_tdparams_folder_path}{os.sep}PL.npy")
        self.PR = np.load(f"{self.outputs_tdparams_folder_path}{os.sep}PR.npy")

        self.stop_event = threading.Event()

        self.probe_located = False

        # Imágenes para trabajo
        self.frame1 = None
        self.frame2 = None
        self.img_punta_izq = None
        self.img_punta_izq = None
        self.imagen_punta = None
        self.pimg1 = None
        self.pimg2 = None

        self.p3d_list = None
        self.cloud_points = None

        self.probe_cloud_points = None
        self.coordenadas_punta = None
        self.puntos_punta_izq = None
        self.puntos_punta_der = None
        self.pred_puntos_punta = None

        self.H_izq_TL = None

        self.logger.info("Se instancia la clase de reconstrucción 3D.")

    def get_cloud_points(self):

        # Se obtienen los puntos ordenados de las dos imágenes y del patrón 3d.
        p2d_list_izq, p2d_list_der, self.p3d_list = gestionTags.ordered_points_list(self.all_synthetic_collection_izq,
                                                                               self.all_synthetic_collection_der)
        # Se le da formato a los puntos.
        points1 = np.array(p2d_list_izq).reshape(-1, 2)
        points2 = np.array(p2d_list_der).reshape(-1, 2)

        points3d = np.array(self.p3d_list).reshape(-1, 3).T

        retval, mask = cv2.findHomography(points1, points2, 0, 1.0)
        mask = mask.ravel()

        # We select only inlier points
        puts1 = points1
        puts2 = points2

        # Se convierten los puntos a coordenadas homogeneas
        puntos1 = processor.cart2hom(puts1.reshape(-1, 2).T)
        puntos2 = processor.cart2hom(puts2.reshape(-1, 2).T)

        self.cloud_points = structure.linear_triangulation(puntos1, puntos2, self.PL, self.PR)
        #tripoints_todos_los_puntos = structure.linear_triangulation(puntos1, puntos2, self.PL, self.PR)
        self.queue_td_points.put(self.cloud_points)

    def save_cloud_points_txt(self):
        if self.cloud_points is not None:
            with open(f'{os.getcwd()}{os.sep}outputs{os.sep}puntos3DEstimados.txt', 'w') as f:
                for i in range(len(self.p3d_list)):
                    if i == len(self.p3d_list) - 1:
                        f.write(
                            f"{self.cloud_points[0][i]},{self.cloud_points[1][i]},{self.cloud_points[2][i]}")

                        f.close
                    else:
                        f.write(
                            f"{self.cloud_points[0][i]},{self.cloud_points[1][i]},{self.cloud_points[2][i]}\n")

    def located_probe_process(self, corners_punta_izq, corners_punta_der):

        # Se crea una lista con los puntos de la punta como un patrón abstracto redimensionado a 10um por pixel.
        puntos_patron_punta = [((0), 0), ((200), 0), ((200), 200), ((0), 200)]

        # Se crea un alista con los puntos de las esquinas del tag de la punta de la imagen izquierda.
        self.puntos_punta_izq = [((corners_punta_izq[(0)]), corners_punta_izq[(1)]),
                            ((corners_punta_izq[(2)]), corners_punta_izq[(3)]),
                            ((corners_punta_izq[(4)]), corners_punta_izq[(5)]),
                            ((corners_punta_izq[(6)]), corners_punta_izq[(7)])]

        # Se crea un alista con los puntos de las esquinas del tag de la punta de la imagen derecha.
        self.puntos_punta_der = [((corners_punta_der[(0)]), corners_punta_der[(1)]),
                            ((corners_punta_der[(2)]), corners_punta_der[(3)]),
                            ((corners_punta_der[(4)]), corners_punta_der[(5)]),
                            ((corners_punta_der[(6)]), corners_punta_der[(7)])]

        # Se calculan las homografías entre las imágenes encontradas de las puntas y el patron abstracto redimensionado a 10um x pixel
        (H_izq, status_izq) = cv2.findHomography(np.array(self.puntos_punta_izq), np.array(puntos_patron_punta), 0,
                                                 3)
        (H_der, status_der) = cv2.findHomography(np.array(self.puntos_punta_der), np.array(puntos_patron_punta), 0,
                                                 3)

        # Se pintan las dos imágenes transformadas.
        self.img_punta_izq = cv2.warpPerspective(self.pimg1.copy(), H_izq, (200, 300))
        self.img_punta_der = cv2.warpPerspective(self.pimg2.copy(), H_der, (200, 300))

        # Se dibuja un punto en el lugar que se estima estaría la punta de la sonda sobre las imágenes transformadas.
        cv2.circle(self.img_punta_izq,
                   (103, 260),
                   radius=2,
                   color=(0, 0, 255), thickness=-1)
        cv2.circle(self.img_punta_der,
                   (123, 260),
                   radius=2,
                   color=(0, 0, 255), thickness=-1)

        # Se hace una transformación inversa para obtener las coordenadas de la punta en las imágenes reales. (Se introduce perspectiva)
        new_pts_1 = np.float32([[103, 260]]).reshape(-1, 1, 2)
        new_pts_2 = np.float32([[123, 260]]).reshape(-1, 1, 2)
        transformados_izq = cv2.perspectiveTransform(new_pts_1, np.linalg.inv(H_izq))
        transformados_der = cv2.perspectiveTransform(new_pts_2, np.linalg.inv(H_der))

        # Se dibujan círculos amarillos en los puntos estimados de la punta de la sonda.
        cv2.circle(self.frame1,
                   (int(transformados_izq[0][0][0]), int(transformados_izq[0][0][1])),
                   radius=2,
                   color=(0, 255, 255), thickness=-1)

        cv2.circle(self.frame2,
                   (int(transformados_der[0][0][0]), int(transformados_der[0][0][1])),
                   radius=2,
                   color=(0, 255, 255), thickness=-1)

        # Se agregan los puntos estimados a la colección de puntos de la punta.
        self.puntos_punta_izq.append(((transformados_izq[0][0][0]), transformados_izq[0][0][1]))
        self.puntos_punta_der.append(((transformados_der[0][0][0]), transformados_der[0][0][1]))

        # Se hace una copia de la imagen de la punta para no trabajar sobre la imagen original.
        self.imagen_punta = self.pimg1.copy()
    def get_probe_cloud_points(self):

        # puntos3d = processor.cart2hom(points3d)

        # points3d = np.array(self.p3d_list, dtype=float32).reshape(-1, 3)

        points1_punta_reales = np.array(self.puntos_punta_izq)
        points2_punta_reales = np.array(self.puntos_punta_der)
        retval, mask = cv2.findHomography(points1_punta_reales, points2_punta_reales, 0, 3.0)
        mask = mask.ravel()
        puts1_punta_reales = points1_punta_reales[mask == 1]
        puts2_punta_reales = points2_punta_reales[mask == 1]
        puntos1_punta_reales = processor.cart2hom(puts1_punta_reales.reshape(-1, 2).T)
        puntos2_punta_reales = processor.cart2hom(puts2_punta_reales.reshape(-1, 2).T)

        self.probe_cloud_points = structure.linear_triangulation(puntos1_punta_reales, puntos2_punta_reales, self.PL,
                                                            self.PR)

        self.coordenadas_punta = [self.probe_cloud_points[0][4], self.probe_cloud_points[1][4], self.probe_cloud_points[2][4]]

        lapso_estimacion_punta = time.time() - self.tiempo_estimacion_punta

        if self.coordenadas_punta[2] < 0 or self.coordenadas_punta[0] < 0 or self.coordenadas_punta[1] < -14000 or \
                self.coordenadas_punta[0] > 14000 or self.coordenadas_punta[1] > 0:

            self.pred_puntos_punta = gestionKalman.prediccion_kalman_sin_correccion(self.kalman_punta,
                                                                               self.measurement_array_punta,
                                                                               self.dt_array_punta,
                                                                               lapso_estimacion_punta)
        else:

            self.pred_puntos_punta = gestionKalman.prediccion_kalman_con_correccion(self.kalman_punta,
                                                                               self.measurement_array_punta,
                                                                               np.array(self.coordenadas_punta),
                                                                               self.dt_array_punta,
                                                                               lapso_estimacion_punta)
            self.tiempo_estimacion_punta = time.time()

        cv2.circle(self.imagen_punta,
                   (int(self.puntos_punta_izq[-1][0] + 10), int(self.puntos_punta_izq[-1][1])),
                   radius=3,
                   color=(0, 0, 255), thickness=-1)

        font2 = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.imagen_punta, str(self.pred_puntos_punta[2]),
                    (int(self.puntos_punta_izq[-1][0] + 10), int(self.puntos_punta_izq[-1][1])),
                    font2, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # H_base, status_base = gestionTags.get_H_from_tags(all_synthetic_collection_izq, tags_collection_patron)
        new_pts_1 = np.float32([[self.pred_puntos_punta[0], self.pred_puntos_punta[1]]]).reshape(-1, 1, 2)
        # print(H_base)
        proyeccion_punta_fondo = cv2.perspectiveTransform(new_pts_1, self.H_izq_TL)

        lapso_estimacion_proyeccion_punta_fondo = time.time() - self.tiempo_estimacion_proyeccion_punta_fondo

        coordenadas_proyeccion_punta_fondo = [proyeccion_punta_fondo[0][0][0], proyeccion_punta_fondo[0][0][1]]
        distancia = 0

        pred_puntos_proyeccion_punta_fondo = gestionKalman.prediccion_kalman_con_correccion(
            self.kalman_proyeccion_punta_fondo, self.measurement_array_proyeccion_punta_fondo,
            np.array(coordenadas_proyeccion_punta_fondo), self.dt_array_proyeccion_punta_fondo,
            lapso_estimacion_proyeccion_punta_fondo)
        self.tiempo_estimacion_proyeccion_punta_fondo = time.time()

        cv2.circle(self.imagen_punta,
                   (int(pred_puntos_proyeccion_punta_fondo[0]), int(pred_puntos_proyeccion_punta_fondo[1])),
                   radius=3,
                   color=(0, 255, 255), thickness=-1)

        cv2.putText(self.imagen_punta, "X: " + str(self.pred_puntos_punta[0]),
                    (int(pred_puntos_proyeccion_punta_fondo[0] + 10),
                     int(pred_puntos_proyeccion_punta_fondo[1])),
                    font2, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.imagen_punta, "Y: " + str(self.pred_puntos_punta[1]),
                    (int(pred_puntos_proyeccion_punta_fondo[0] + 10),
                     int(pred_puntos_proyeccion_punta_fondo[1] + 20)),
                    font2, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
    def save_probe_cloud_pints_txt(self):

        with open(f'{os.getcwd()}{os.sep}outputs{os.sep}puntos3DPatron.txt', 'w') as f:
            for i in range(len(self.p3d_list)):
                if i == len(self.p3d_list) - 1:
                    f.write(f"{self.p3d_list[i][0]},{self.p3d_list[i][1]},{self.p3d_list[i][2]}\n")

                    f.write(
                        f"{self.probe_cloud_points[0][0]},{self.probe_cloud_points[1][0]},{self.probe_cloud_points[2][0]}\n")
                    f.write(
                        f"{self.probe_cloud_points[0][1]},{self.probe_cloud_points[1][1]},{self.probe_cloud_points[2][1]}\n")
                    f.write(
                        f"{self.probe_cloud_points[0][2]},{self.probe_cloud_points[1][2]},{self.probe_cloud_points[2][2]}\n")
                    f.write(
                        f"{self.probe_cloud_points[0][3]},{self.probe_cloud_points[1][3]},{self.probe_cloud_points[2][3]}\n")

                    f.write(f"{self.probe_cloud_points[0][4]},{self.probe_cloud_points[1][4]},{self.probe_cloud_points[2][4]}")
                    f.close
                else:
                    f.write(f"{self.p3d_list[i][0]},{self.p3d_list[i][1]},{self.p3d_list[i][2]}\n")

        self.xs.append(round(self.pred_puntos_punta[0][0]))
        self.ys.append(round(self.pred_puntos_punta[1][0]))
        self.zs.append(round(self.pred_puntos_punta[2][0]))
        self.tiempos.append(time.time())
        self.prom_xs.append(round(sum(self.xs) / len(self.xs)))
        self.prom_ys.append(round(sum(self.ys) / len(self.ys)))
        self.prom_zs.append(round(sum(self.zs) / len(self.zs)))
        if len(self.xs) > 1:
            self.xs.pop(0)
            self.ys.pop(0)
            self.zs.pop(0)
            self.tiempos.pop(0)
            self.prom_xs.pop(0)
            self.prom_ys.pop(0)
            self.prom_zs.pop(0)

        if len(self.xs) == 1:
            with open(f'{os.getcwd()}{os.sep}outputs{os.sep}listaMediciones.txt', 'w') as f:
                f.write(str(self.tiempos[0]) + ',' + str(self.xs[0]) + ',' + str(self.ys[0]) + ',' + str(
                    self.zs[0]) + ',' + str(
                    self.prom_xs[0]) + ',' + str(self.prom_ys[0]) + ',' + str(self.prom_zs[0]))
                f.close()
            # os.system("graficasPlt.py")

        if len(self.xs) > 1:
            lista_mediciones = []
            for i in range(len(self.xs)):
                lista_mediciones.append(
                    str(self.tiempos[i]) + ',' + str(self.xs[i]) + ',' + str(self.ys[i]) + ',' + str(
                        self.zs[i]) + ',' + str(
                        self.prom_xs[i]) + ',' + str(self.prom_ys[i]) + ',' + str(self.prom_zs[i]))

            with open(f'{os.getcwd()}{os.sep}outputs{os.sep}listaMediciones.txt', 'w') as f:
                for line in lista_mediciones:
                    f.write('\n' + line)
                f.close

    def execute(self):

        self.logger.debug("Se ingresa en hilo de reconstrucción 3D")

        while not self.stop_event.is_set():

            # Se leen los frames
            ret1, frm1 = self.cap_left.read()
            ret2, frm2 = self.cap_right.read()

            cv2.imshow("img_izq", frm1)
            cv2.imshow("img_der", frm2)

            # Se hacen copias de los frames para trabajar sobre ellas y no sobre las imágenes originales
            if self.cam_calib is True:
                self.pimg1 = cv2.undistort(frm1.copy(), self.K1, self.D1, self.K1)
                self.pimg2 = cv2.undistort(frm2.copy(), self.K2, self.D2, self.K2)
            else:
                # Opcional para trabajar sobre imagenes no corregidas
                self.pimg1 = frm1.copy()
                self.pimg2 = frm2.copy()

            self.frame1 = self.pimg1.copy()
            self.frame2 = self.pimg2.copy()

            # Se buscan los tags en las imágenes
            self.tags_collection_izq = gestionTags.find_tags_subpix(self.pimg1, self.tags_collection_izq)
            self.tags_collection_der = gestionTags.find_tags_subpix(self.pimg2, self.tags_collection_der)

            # Con base en los tags encontrados y en los tags patronados, se hace una estimación de todos los puntos que debería contener cada una de las imágenes.
            self.all_synthetic_collection_izq, self.H_izq_TL, H_izq_BL = gestionTags.get_all_synthetic_points_tags(
                self.tags_collection_izq,
                self.tags_collection_patron,
                self.all_synthetic_collection_izq)
            self.all_synthetic_collection_der, H_der_TL, H_der_BL = gestionTags.get_all_synthetic_points_tags(
                self.tags_collection_der,
                self.tags_collection_patron,
                self.all_synthetic_collection_der)

            try:
                # Se obtienen los puntos organizados en forma de lista
                puntos_frame_1, puntos_frame_2 = gestionTags.get_listado_tags(self.all_synthetic_collection_izq,
                                                                              self.all_synthetic_collection_der, synthetic=True)
            except Exception as e:
                self.logger.warning(f"No se encuentran tags suficientes en las imágenes. Error: {e}")
                continue

            # puntos_frame_1, puntos_frame_2 = gestionTags.get_listado_tags(tags_collection_izq,
            # tags_collection_der, synthetic=False)
            # Se pintan círculos sobre los puntos hallados y estimados
            for i in range(0, len(puntos_frame_1), 1):
                cv2.circle(self.frame1,
                           (int(puntos_frame_1[i][0]), int(puntos_frame_1[i][1])),
                           radius=2,
                           color=(255, 0, 0), thickness=-1)
                cv2.circle(self.frame2,
                           (int(puntos_frame_2[i][0]), int(puntos_frame_2[i][1])),
                           radius=2,
                           color=(255, 0, 0), thickness=-1)

            # Se extraen las esquinas del patrón de la punta en un arreglo independiente
            corners_punta_izq = np.array(self.all_synthetic_collection_izq['15BL']['corners']).flatten()
            corners_punta_der = np.array(self.all_synthetic_collection_der['15BL']['corners']).flatten()

            if None not in corners_punta_der or None not in corners_punta_izq:
                self.probe_located = True
                # print(corners_punta_izq)
                # print(corners_punta_der)
                self.logger.debug("Punta localizada")

            else:
                self.logger.warning("Punta no encontrada en las imágenes")
            try:
                try:
                    self.get_cloud_points()
                    self.save_cloud_points_txt()
                    self.imagen_punta = self.frame1
                except Exception as e:
                    self.looger.error(f"Error obteniendo nube de puntos de la base. Error: {e}")

                if self.probe_located is True and self.PL is not None and self.PR is not None and self.H_izq_TL is not None:
                    self.logger.info(f"Punta localizada y matrices PL y PR presentes")
                    try:
                        self.located_probe_process(corners_punta_izq, corners_punta_der)
                        self.get_probe_cloud_points()
                        self.save_probe_cloud_pints_txt()
                    except Exception as e:
                        self.logger.error(f"Error obteniendo y guardando nube de puntos de la punta. Error: {e}")

            except Exception as e:
                self.logger.error(f"Error en execute:  {e}")
                # img_stit, vis, pts1, pts2 = stitching_images(frame1, frame2, np.array(puntos_frame_1),
                #                                            np.array(puntos_frame_2))
                # img_stit = frame1
                # vis = frame2
                self.img_punta_izq = self.frame1
                self.img_punta_der = self.frame2
                self.imagen_punta = self.frame1
                # img_tags_r = frame2

            cv2.imshow("Frame 1", self.frame1)
            cv2.imshow("Imagen Punta", self.imagen_punta)
            cv2.imshow("Frame 2", self.frame2)
            # cv2.imshow("IZQ", self.img_punta_izq)
            # cv2.imshow("Imagen_DER", self.img_punta_der)
            # cv2.imshow("Puntos", vis)
            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key == ord('q'):
                # self.cap_left.release()
                # self.cap_right.release()
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        # self.cap_left.release()
        # self.cap_right.release()

    def stop(self):
        self.stop_event.set()
