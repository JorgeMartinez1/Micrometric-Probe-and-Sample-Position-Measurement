import logging
import serial
import serial.tools.list_ports
import sys
import os
import os.path
import pyvisa as visa
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import cv2
import numpy as np
import threading
from queue import Queue
import queue
import matplotlib.pyplot as plt

from gui.td_controls_window import TDControlsWindow
from core.micrometric_scenes_td_reconstruction import MicroEscenesReconstruction
from core.grafica_reconstructed_td import GraphTDTk
from core.serial_control import SerialControl
from core.video_devices import VideoDevices


class TDControl(TDControlsWindow):
    def __init__(self, logger: logging = None, caps:VideoDevices = None, ser:SerialControl=None):
        super().__init__()
        self.logger = logger
        self.caps = caps
        self.cap_left = self.caps.cap_left
        self.cap_right = self.caps.cap_right
        self.ser = ser

        if self.cap_left is not None and self.cap_right is not None:
            self.combo_cam_right.config(text=str(self.cap_right.get(cv2.CAP_PROP_POS_FRAMES)))
            self.combo_cam_left.config(text=str(self.cap_left.get(cv2.CAP_PROP_POS_FRAMES)))

        self.logger.info("Iniciando control para reconstrucción de escenas 3D.")

        self.outputs_folder_path = f"{os.getcwd()}{os.sep}outputs"
        self.outputs_tdparams_folder_path = f"{self.outputs_folder_path}{os.sep}tdparams"

        self.k_l = None
        self.k_r = None
        self.d_l = None
        self.d_r = None

        self.img_tags_base = None
        self.img_tag_punta = None

        self.left_p_matrix = None
        self.right_p_matrix = None

        self.logger.debug("Verificando existencia de folders de salida")
        self.folders_verify = False
        self.verificar_folders_outputs()

        self.logger.debug("Verificando existencia de patrones de referencia")
        self.patterns_verify = False
        self.verificar_patterns()

        self.logger.debug("Verificando existencia de parámetros K y D de calibración de cámara")
        self.cam_params_verify = False
        self.verificar_cam_params()

        self.logger.debug("Verificando existencia de matrices de proyectividad P")
        self.p_matrix_verify = False
        self.verificar_p_matrix()

        self.stop_thread = False

        self.reconstruction = threading.Thread()
        self.reconstruction_instance = None

        self.graph_3d_plt_instance = None

        self.queue_probe_location = Queue()
        self.queue_td_points = Queue()

    def verificar_p_matrix(self):
        list_p_matrix = []
        list_p_matrix.append(f"{self.outputs_tdparams_folder_path}{os.sep}PL.npy")
        list_p_matrix.append(f"{self.outputs_tdparams_folder_path}{os.sep}PR.npy")

        for p_matrix in list_p_matrix:

            if os.path.exists(p_matrix):
                self.logger.info(f"El archivo '{p_matrix}' existe.")
                self.p_matrix_verify = True
            else:
                self.logger.warning(f"El archivo '{p_matrix}' no existe. Asegúrese de agregarlo en la ruta indicada.")
                self.p_matrix_verify = False
                break

            if p_matrix == f"{self.outputs_tdparams_folder_path}{os.sep}PL.npy":
                try:
                    # Cargar el archivo .npy en una variable
                    self.left_p_matrix = np.load(p_matrix)
                    self.logger.debug("Se carga la matriz P de la cámara izquierda")
                except FileNotFoundError:
                    self.logger.critical(f"El archivo '{p_matrix}' no fue encontrado.")

            if p_matrix == f"{self.outputs_tdparams_folder_path}{os.sep}PR.npy":
                try:
                    # Cargar el archivo .npy en una variable
                    self.right_p_matrix = np.load(p_matrix)
                    self.logger.debug("Se carga la matriz P de la cámara derecha")
                except FileNotFoundError:
                    self.logger.warning(f"El archivo '{p_matrix}' no fue encontrado.")

    def verificar_cam_params(self):
        list_cam_params = []
        list_cam_params.append(f"{self.outputs_tdparams_folder_path}{os.sep}kl.npy")
        list_cam_params.append(f"{self.outputs_tdparams_folder_path}{os.sep}kr.npy")
        list_cam_params.append(f"{self.outputs_tdparams_folder_path}{os.sep}dl.npy")
        list_cam_params.append(f"{self.outputs_tdparams_folder_path}{os.sep}dr.npy")

        for archivo in list_cam_params:
            if os.path.exists(archivo):
                self.logger.info(f"El archivo '{archivo}' existe.")
                self.cam_params_verify = True
            else:
                self.logger.warning(f"El archivo '{archivo}' no existe. Asegúrese de agregarlo en la ruta indicada.")
                self.cam_params_verify = False
                break

            if archivo == f"{self.outputs_tdparams_folder_path}{os.sep}kl.npy":
                try:
                    # Cargar el archivo .npy en una variable
                    self.k_l = np.load(archivo)
                    self.logger.debug("Se carga el parámetro K_L")
                except FileNotFoundError:
                    self.logger.warning(f"El archivo '{archivo}' no fue encontrado.")

            if archivo == f"{self.outputs_tdparams_folder_path}{os.sep}kr.npy":
                try:
                    # Cargar el archivo .npy en una variable
                    self.k_r = np.load(archivo)
                    self.logger.debug("Se carga el parámetro K_R")
                except FileNotFoundError:
                    self.logger.warning(f"El archivo '{archivo}' no fue encontrado.")

            if archivo == f"{self.outputs_tdparams_folder_path}{os.sep}dl.npy":
                try:
                    # Cargar el archivo .npy en una variable
                    self.d_l = np.load(archivo)
                    self.logger.debug("Se carga el parámetro D_L")
                except FileNotFoundError:
                    self.logger.warning(f"El archivo '{archivo}' no fue encontrado.")

            if archivo == f"{self.outputs_tdparams_folder_path}{os.sep}dr.npy":
                try:
                    # Cargar el archivo .npy en una variable
                    self.d_r = np.load(archivo)
                    self.logger.debug("Se carga el parámetro D_R")
                except FileNotFoundError:
                    self.logger.warning(f"El archivo '{archivo}' no fue encontrado.")

    def verificar_patterns(self):
        if os.path.exists(f"{self.outputs_tdparams_folder_path}{os.sep}tableroBase12Tags.jpg"):
            self.logger.info(f"El archivo '{self.outputs_tdparams_folder_path}{os.sep}tableroBase12Tags.jpg' existe.")
            self.patterns_verify = True
        else:
            self.logger.critical(f"El archivo '{self.outputs_tdparams_folder_path}{os.sep}tableroBase12Tags.jpg' no existe. Asegúrese de agregarlo en la ruta indicada.")
            self.patterns_verify = False
            return

        if os.path.exists(f"{self.outputs_tdparams_folder_path}{os.sep}tagPunta.png"):
            self.logger.info(f"El archivo '{self.outputs_tdparams_folder_path}{os.sep}tagPunta.png' existe.")
            self.patterns_verify = True
        else:
            self.logger.critical(f"El archivo '{self.outputs_tdparams_folder_path}{os.sep}tagPunta.png' no existe. Asegúrese de agregarlo en la ruta indicada.")
            self.patterns_verify = False

    def verificar_folders_outputs(self):

        if not os.path.exists(self.outputs_folder_path):
            self.logger.warning(f"Folder {self.outputs_folder_path} no existe.")
            self.folders_verify = False
            try:
                self.logger.debug(f"Intentando crear folder {self.outputs_folder_path}.")
                os.makedirs(self.outputs_folder_path)
                self.logger.info(f"Directorio '{self.outputs_folder_path}' creado exitosamente.")
                self.folders_verify = True

            except OSError as e:
                self.logger.critical(f"No fue posible crear el folder {self.outputs_folder_path}. Por favor verifique que la aplicación cuenta con los permisos necesarios. Puede intentar ejecutarla en modo 'Administrador'. Error: {e}")
                # print(f"Error al crear el directorio '{self.outputs_folder_path}': {e}")
                self.folders_verify = False
        else:
            self.logger.info(f"El directorio '{self.outputs_folder_path}' ya existe.")
            self.folders_verify = True

        if not os.path.exists(self.outputs_tdparams_folder_path):
            self.logger.warning(f"Folder {self.outputs_tdparams_folder_path} no existe.")
            self.folders_verify = False
            try:
                self.logger.debug(f"Intentando crear folder {self.outputs_tdparams_folder_path}.")
                os.makedirs(self.outputs_tdparams_folder_path)
                self.logger.info(f"Directorio '{self.outputs_tdparams_folder_path}' creado exitosamente.")
                self.folders_verify = True

            except OSError as e:
                self.logger.critical(f"No fue posible crear el folder {self.outputs_tdparams_folder_path}. Por favor verifique que la aplicación cuenta con los permisos necesarios. Puede intentar ejecutarla en modo 'Administrador'. Error: {e}")
                self.folders_verify = False
        else:
            self.logger.info(f"El directorio '{self.outputs_tdparams_folder_path}' ya existe.")
            self.folders_verify = True

    def launch_simulation_saved_videos(self):
        verify_videos = False
        verify_videos = self.verify_videos_simulation()
        if verify_videos is False:
            self.logger.warning('No es posible hacer la simulación ya que no se cuenta con los videos.')
            return
        try:
            # Cargar el video izquierdo
            self.cap_left = cv2.VideoCapture(f"{self.outputs_tdparams_folder_path}{os.sep}video_l.avi")
            self.cap_right = cv2.VideoCapture(f"{self.outputs_tdparams_folder_path}{os.sep}video_r.avi")

            self.logger.debug("Se cargan los videos para simulación")
        except FileNotFoundError:
            self.logger.critical(f"No fue posible cargar los videos. Asegúrese de tener los controladores de video .avi instalados en el equipo.")
        print('Simulación con videos')

    def verify_videos_simulation(self):
        list_videos_simulation = []
        list_videos_simulation.append(f"{self.outputs_tdparams_folder_path}{os.sep}video_l.avi")
        list_videos_simulation.append(f"{self.outputs_tdparams_folder_path}{os.sep}video_r.avi")

        for video_simulation in list_videos_simulation:

            if os.path.exists(video_simulation):
                self.logger.info(f"El archivo '{video_simulation}' existe.")
                return True
            else:
                self.logger.warning(f"El archivo '{video_simulation}' no existe. Asegúrese de agregarlo en la ruta indicada.")
                return False

    def fn_gen_3d(self):

        if self.folders_verify is False or self.patterns_verify is False or self.p_matrix_verify is False:
            self.logger.error(f"Los folder de salida o los parámetros iniciales necesarios para la operación no existen. Por favor verifique que la aplicación cuente con permisos para crear o administarar estos o genérelos de ser necesario.")
            return

        if self.cap_left is None or self.cap_right is None:
            self.logger.debug("Lanzando simulación con videos guardados")
            self.launch_simulation_saved_videos()
            self.img_3d_thread()
        elif self.cap_left == 'None' or self.cap_right == 'None':
            self.logger.debug("Lanzando simulación con videos guardados")
            self.launch_simulation_saved_videos()
            self.img_3d_thread()

        else:
            self.img_3d_thread()

    def fn_stop_thread(self):
        # Para detener el hilo cuando sea necesario:
        self.reconstruction_instance.stop()
        self.reconstruction.join()  # Espera a que el hilo termine
        self.btn_both_cams.grid_remove()

    def img_3d_thread(self):
        self.logger.debug("Lanzando nuevo hilo para reconstrucción 3D.")

        self.btn_both_cams = ttk.Button(self.frame_buttons, text="Detener Proceso", command=self.fn_stop_thread)
        self.btn_both_cams.grid(row=1, column=0, padx=5, pady=5)

        if sys.platform == "linux" or sys.platform == "linux2":
            self.stop_thread = False
            self.logger.debug("Se identifica el sistema operativo Windows.")
            self.reconstruction_instance = MicroEscenesReconstruction(logger=self.logger, caps=self.caps,
                                                                      configuracion=(16, 1),
                                                                      path_params=self.outputs_tdparams_folder_path,
                                                                      cam_calib=False, qtdp=self.queue_td_points)
            self.reconstruction = threading.Thread(target=self.reconstruction_instance.execute, args=())
            self.reconstruction.start()

            self.verificar_cola()
        # linux
        elif sys.platform == "darwin" or sys.platform == "Darwin":
            self.stop_thread = False
            self.logger.debug("Se identifica el sistema operativo Windows.")
            self.reconstruction_instance = MicroEscenesReconstruction(logger=self.logger, caps=self.caps,
                                                                      configuracion=(16, 1),
                                                                      path_params=self.outputs_tdparams_folder_path,
                                                                      cam_calib=False, qtdp=self.queue_td_points)
            self.reconstruction = threading.Thread(target=self.reconstruction_instance.execute, args=())
            self.reconstruction.start()

            self.verificar_cola()
        # OS X
        elif sys.platform == "win32":
            self.stop_thread = False
            self.logger.debug("Se identifica el sistema operativo Windows.")
            self.reconstruction_instance = MicroEscenesReconstruction(logger=self.logger, caps=self.caps,
                                                                      configuracion=(16, 1),
                                                                 path_params=self.outputs_tdparams_folder_path,
                                                                 cam_calib= False, qtdp= self.queue_td_points)
            self.reconstruction = threading.Thread(target=self.reconstruction_instance.execute, args=())
            self.reconstruction.start()

            self.verificar_cola()

        else:
            self.logger.debug(f"Aún no hay soporte para el sistema operativo {sys.platform}. Por favor escriba su comentario por Github.")
    def verificar_cola(self):
        try:
            # Obtener un mensaje de la cola (esto bloqueará si la cola está vacía)
            mensaje = self.queue_td_points.get_nowait()

            if mensaje is None:
                # El hilo secundario ha finalizado
                return

            # Realizar alguna acción en el hilo principal basada en el mensaje
            self.logger.info(f"Mensaje recibido en el hilo principal: {mensaje}")

            # Puedes agregar más lógica aquí según tus necesidades

        except queue.Empty:
            pass  # La cola está vacía, no hay nuevos datos

        # Programar la próxima verificación después de un intervalo de tiempo
        self.after(500, self.verificar_cola)
    def on_closing(self):
        # Esta función se ejecutará al cerrar la ventana
        if self.reconstruction_instance is not None:
            self.fn_stop_thread()
            # self.logger.warning("Cerrando hilos activos.")
        # Agrega aquí cualquier lógica adicional que desees ejecutar antes de cerrar la ventana
        self.destroy()


def main():
    app = TDControl()
    app.mainloop()


# ==========================================================================

if __name__ == '__main__':
    main()