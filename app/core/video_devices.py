import logging
import sys
import os.path
import time
import cv2

from gui.video_window import VideoWindow

class VideoDevices:
    def __init__(self, logger: logging = None):
        super().__init__()
        self.logger = logger

        self.cap_left = cv2.VideoCapture
        self.label_cap_left = ''
        self.video_window_left= None
        self.cap_right = cv2.VideoCapture
        self.label_cap_right = ''
        self.video_window_right = None
        self.cap_3 = cv2.VideoCapture
        self.label_cap_3 = ''
        self.video_window_3 = None
        self.cap_4 = cv2.VideoCapture
        self.label_cap_4 = ''
        self.video_window_4 = None

        self.cam_list = []

        self.logger.info("Se inicializa la clase para control de dispositivos de video")

    def get_cam_list(self):
        "Este método itera entre los primeros 20 dispositivos de video conectados al sistema y los regresa como un listado."
        self.logger.debug("Buscando cámaras disponibles en el equipo")
        self.cam_list = []
        local_cam_list = []
        for i in range(20):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                local_cam_list.append(str(i))
                self.logger.info('Se encontró la cámara ' + str(i))
            else:
                break
        if len(local_cam_list) == 0:
            self.logger.warning("No se encontraron cámaras disponibles en el equipo.")

        self.logger.debug("Verificando si hay videos disponibles para simulación.")

        platform = sys.platform
        if platform.startswith('win'):
            path_video_l = f"{os.getcwd()}\\outputs\\tdparams\\video_l.avi"
            path_video_r = f"{os.getcwd()}\\outputs\\tdparams\\video_r.avi"
        elif platform.startswith('linux'):
            path_video_l = f"{os.getcwd()}/outputs/tdparams/video_l.avi"
            path_video_r = f"{os.getcwd()}/outputs/tdparams/video_r.avi"
        elif platform.startswith('darwin'):
            path_video_l = f"{os.getcwd()}/outputs/tdparams/video_l.avi"
            path_video_r = f"{os.getcwd()}/outputs/tdparams/video_r.avi"
        else:
            self.logger.critical(f'Sistema operativo desconocido {str(platform)}')
            return []

        if os.path.exists(path_video_l) and os.path.exists(path_video_r):
            self.logger.debug("Agregando videos encontrados.")
            local_cam_list.append('video_l.avi')
            local_cam_list.append('video_r.avi')
        else:
            self.logger.warning("No se encontrearon videos para simulación")

        self.cam_list = local_cam_list
        return local_cam_list

    def set_config_cameras(self, label_left, label_right):
        """ Método para establecer la configuración de las cámaras para estereoscopía"""
        if self.video_window_left is not None and self.video_window_left.winfo_exists():
            self.video_window_left.destroy()
            self.cap_left.release()

        if self.video_window_right is not None and self.video_window_right.winfo_exists():
            self.video_window_right.destroy()
            self.cap_right.release()

        self.cap_left = cv2.VideoCapture(int(label_left))
        self.label_cap_left = label_left

        self.cap_right = cv2.VideoCapture(int(label_right))
        self.label_cap_right = label_right

    def set_config_videos(self, label_left, label_right):
        """ Método para establecer la configuración de las cámaras para estereoscopía"""
        if self.video_window_left is not None and self.video_window_left.winfo_exists():
            self.video_window_left.destroy()
            self.cap_left.release()

        if self.video_window_right is not None and self.video_window_right.winfo_exists():
            self.video_window_right.destroy()
            self.cap_right.release()

        platform = sys.platform
        if platform.startswith('win'):
            path_videos = f"{os.getcwd()}\\outputs\\tdparams\\"
        elif platform.startswith('linux'):
            path_videos = f"{os.getcwd()}/outputs/tdparams/"
        elif platform.startswith('darwin'):
            path_videos = f"{os.getcwd()}/outputs/tdparams/"
        else:
            self.logger.critical(f'Sistema operativo desconocido {str(platform)}')
            return

        if os.path.exists(f"{path_videos}video_l.avi") and os.path.exists(f"{path_videos}video_r.avi"):
            self.label_cap_left = f"{path_videos}video_l.avi"
            self.logger.debug("Se asigna video izquierdo para simulación.")
            self.cap_left = cv2.VideoCapture(f"{path_videos}video_l.avi")

            self.label_cap_right = f"{path_videos}video_r.avi"
            self.logger.debug("Se asigna video derecho para simulación.")
            self.cap_right = cv2.VideoCapture(f"{path_videos}video_r.avi")

    def show_left_cam(self, label_cam:str = None):
        if self.video_window_left is not None and self.video_window_left.winfo_exists():
            self.video_window_left.destroy()
            self.cap_left.release()
            self.logger.warning("Cerrando ventana anterior")

        if label_cam is None or label_cam == '':
            self.logger.error("No se ha especificado la etiqueta de la cámara o el video")
            return
        elif label_cam == 'video_r.avi':
            self.logger.warning("Está asignando el video derecho a la cámara izquierda. Corrija esto e inténtelo de nuevo")
            return
        elif label_cam == 'video_l.avi':
            platform = sys.platform
            if platform.startswith('win'):
                path_video_l = f"{os.getcwd()}\\outputs\\tdparams\\{label_cam}"
            elif platform.startswith('linux'):
                path_video_l = f"{os.getcwd()}/outputs/tdparams/{label_cam}"
            elif platform.startswith('darwin'):
                path_video_l = f"{os.getcwd()}/outputs/tdparams/{label_cam}"
            else:
                self.logger.critical(f'Sistema operativo desconocido {str(platform)}')
                return []

            if os.path.exists(path_video_l):
                self.label_cap_left = path_video_l
                self.logger.debug("Se asigna video izquierdo para simulación.")
                self.cap_left = cv2.VideoCapture(path_video_l)
        else:
            self.label_cap_left = label_cam
            self.cap_left = cv2.VideoCapture(int(label_cam))
            self.logger.debug(f"Se asigna como cámara izquierda {label_cam}")

        try:
            self.video_window_left = VideoWindow('CAMERA LEFT')
            self.video_window_left.execute_video(self.cap_left)
            self.video_window_left.mainloop()
        except Exception as e:
            self.logger.critical(f"'Error leyendo cámara: {label_cam}'. Error: {str(e)}")
            return

    def show_right_cam(self, label_cam:str = None):
        if self.video_window_right is not None and self.video_window_right.winfo_exists():
            self.video_window_right.destroy()
            self.cap_right.release()
            self.logger.warning("Cerrando ventana anterior")

        if label_cam is None or label_cam == '':
            self.logger.error("No se ha especificado la etiqueta de la cámara o el video")
            return
        elif label_cam == 'video_l.avi':
            self.logger.warning("Está asignando el video izquierdo a la cámara derecha. Corrija esto e inténtelo de nuevo")
            return
        elif label_cam == 'video_r.avi':
            platform = sys.platform
            if platform.startswith('win'):
                path_video_r = f"{os.getcwd()}\\outputs\\tdparams\\{label_cam}"
            elif platform.startswith('linux'):
                path_video_r = f"{os.getcwd()}/outputs/tdparams/{label_cam}"
            elif platform.startswith('darwin'):
                path_video_r = f"{os.getcwd()}/outputs/tdparams/{label_cam}"
            else:
                self.logger.critical(f'Sistema operativo desconocido {str(platform)}')
                return []

            if os.path.exists(path_video_r):
                self.label_cap_right = path_video_r
                self.logger.debug("Se asigna video derecho para simulación.")
                self.cap_right = cv2.VideoCapture(path_video_r)
        else:
            self.label_cap_right = label_cam
            self.cap_right = cv2.VideoCapture(int(label_cam))
            self.logger.debug(f"Se asigna como cámara derecha {label_cam}")

        try:
            self.video_window_right = VideoWindow('CAMERA RIGHT')
            self.video_window_right.execute_video(self.cap_right)
            self.video_window_right.mainloop()
        except Exception as e:
            self.logger.critical(f"'Error leyendo cámara: {label_cam}'. Error: {str(e)}")
            return

    def show_3_cam(self, label_cam:str = None):
        if self.video_window_3 is not None and self.video_window_3.winfo_exists():
            self.video_window_3.destroy()
            self.cap_3.release()
            self.logger.warning("Cerrando ventana anterior")
        if label_cam is None or label_cam == '':
            self.logger.error("No se ha especificado la etiqueta de la cámara")
            return
        elif label_cam == 'video_l.avi' or label_cam == 'video_r.avi':
            self.logger.warning("No puede asignar un video a la cámara desoporte 3")
            return
        else:
            self.label_cap_3 = label_cam
            self.cap_right = cv2.VideoCapture(int(label_cam))
            self.logger.debug(f"Se asigna como cámara de soporte {label_cam}")

        try:
            self.video_window_3 = VideoWindow('CAMERA SUPPORT 3')
            self.video_window_3.execute_video(self.cap_3)
            self.video_window_3.mainloop()
        except Exception as e:
            self.logger.critical(f"'Error leyendo cámara: {label_cam}'. Error: {str(e)}")
            return

    def show_4_cam(self, label_cam:str = None):
        if self.video_window_4 is not None and self.video_window_4.winfo_exists():
            self.video_window_4.destroy()
            self.cap_4.release()
            self.logger.warning("Cerrando ventana anterior")
        if label_cam is None or label_cam == '':
            self.logger.error("No se ha especificado la etiqueta de la cámara")
            return
        elif label_cam == 'video_l.avi' or label_cam == 'video_r.avi':
            self.logger.warning("No puede asignar un video a la cámara desoporte 4")
            return
        else:
            self.label_cap_4 = label_cam
            self.cap_right = cv2.VideoCapture(int(label_cam))
            self.logger.debug(f"Se asigna como cámara de soporte {label_cam}")

        try:
            self.video_window_4 = VideoWindow('CAMERA SUPPORT 4')
            self.video_window_4.execute_video(self.cap_4)
            self.video_window_4.mainloop()
        except Exception as e:
            self.logger.critical(f"'Error leyendo cámara: {label_cam}'. Error: {str(e)}")
            return

    def close_all_video_windows(self):
        """Método para cerrar todas las ventanas de video activas."""

        if self.video_window_left is not None and self.video_window_left.winfo_exists():
            self.video_window_left.destroy()
        if self.video_window_right is not None and self.video_window_right.winfo_exists():
            self.video_window_right.destroy()
        if self.video_window_3 is not None and self.video_window_3.winfo_exists():
            self.video_window_3.destroy()
        if self.video_window_4 is not None and self.video_window_4.winfo_exists():
            self.video_window_4.destroy()