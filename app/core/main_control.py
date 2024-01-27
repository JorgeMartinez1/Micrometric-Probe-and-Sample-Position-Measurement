import tkinter as tk
from tkinter import ttk, scrolledtext
import queue
import logging
import cv2
import threading

from gui.main_window import MainWindow

from core.td_controls import TDControl
from core.serial_control import SerialControl
from core.video_devices import VideoDevices
import time


logger = logging.getLogger('mmunam')
logger.setLevel(logging.DEBUG)


class QueueHandler(logging.Handler):
    """Class to send logging records to a queue

    It can be used from different threads
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.line_counter = 1

    def emit(self, record):
        record.lineno = self.line_counter
        self.line_counter += 1
        self.log_queue.put(record)


class MainControl(MainWindow):
    def __init__(self):
        super().__init__()

        # Create a logging handler using a queue
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s [Line:%(lineno)d]: %(message)s')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self.after(10, self.poll_log_queue)

        logger.info('------------- Inicio -----------')

        self.ser = SerialControl(logger)
        self.caps = VideoDevices(logger)

        self.td_reconstruct = None

        self.fn_listar_camaras_disponibles()

    def on_combobox_cam_left_change(self):
        nuevo_valor = self.combo_cam_left.get()
        print(f"Nuevo valor seleccionado: {nuevo_valor}")

    def fn_listar_camaras_disponibles(self):
        cam_list = self.caps.get_cam_list()
        self.combo_cam_right['values'] = cam_list
        self.combo_cam_left['values'] = cam_list

    def display(self, record):
        msg = self.queue_handler.format(record)
        self.text_area.configure(state='normal')
        self.text_area.insert(tk.END, msg + '\n', record.levelname)
        self.text_area.configure(state='disabled')
        # Autoscroll to the bottom
        self.text_area.yview(tk.END)

    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)

        # self.poll_log_queue()
        self.after(100, self.poll_log_queue)

    def fn_ver_cam_left(self):
        cam_str = self.combo_cam_left.get()
        logger.debug("Se lanza la cámara izquierda con id: " + cam_str)

        if cam_str == 'No se encuentran cámaras' or cam_str == '' or cam_str == 'None':
            logger.warning('No ha seleccionado una cámara')
            return
        else:
            self.caps.show_left_cam(label_cam=cam_str)

    def fn_ver_cam_right(self):
        cam_str = self.combo_cam_right.get()
        logger.debug("Se lanza la cámara derecha con id: " + cam_str)

        if cam_str == 'No se encuentran cámaras' or cam_str == '' or cam_str == 'None':
            logger.warning('No ha seleccionado una cámara')
            return
        else:
            self.caps.show_right_cam(label_cam=cam_str)

    def fn_gen_3d(self):

        cam_str_right = self.combo_cam_right.get()
        cam_str_left = self.combo_cam_left.get()

        if cam_str_right == 'No se encuentran cámaras' or cam_str_right == '' or cam_str_right == 'None':
            logger.warning('No ha seleccionado una cámara derecha o video para simulación.')
            return

        if cam_str_left == 'No se encuentran cámaras' or cam_str_left == '' or cam_str_left == 'None':
            logger.warning('No ha seleccionado una cámara izquierda o video para simulación.')
            return

        if cam_str_left == 'video_l.avi' and cam_str_right != 'video_r.avi':
            logger.warning("Para simulaciones, debe seleccionar los dos videos como fuente")
            return

        if cam_str_left != 'video_l.avi' and cam_str_right == 'video_r.avi':
            logger.warning("Para simulaciones, debe seleccionar los dos videos como fuente")
            return

        if cam_str_left == 'video_r.avi' or cam_str_right == 'video_l.avi':
            logger.warning("Asigne los videos correctamente a las fuentes izquierda y derecha")
            return

        if cam_str_left == 'video_l.avi' or cam_str_right == 'video_r.avi':
            self.caps.set_config_videos(cam_str_left, cam_str_right)
            logger.debug("Se configuran los videos para simulación.")
        else:
            self.caps.set_config_cameras(cam_str_left, cam_str_right)

        self.td_reconstruct = TDControl(logger, self.caps, self.ser)
        self.td_reconstruct.mainloop()

    def fn_pictures_reports(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                     "jorge.martinez@icat.unam.mx")

    def fn_connect_vna_pna(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                     "jorge.martinez@icat.unam.mx")

    def fn_get_sample(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                     "jorge.martinez@icat.unam.mx")
    def fn_open_gen_patterns(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                      "jorge.martinez@icat.unam.mx")

    def fn_calib_cam(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                      "jorge.martinez@icat.unam.mx")

    def fn_ambas_camaras(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                      "jorge.martinez@icat.unam.mx")

    def fn_probe_location(self):
        logger.info("Para obtener el software completo, comuníquese con los desarrolladores a "
                      "jorge.martinez@icat.unam.mx")

    def on_closing(self):
        # Esta función se ejecutará al cerrar la ventana
        self.ser.close_all()

        if self.td_reconstruct is not None and self.td_reconstruct.winfo_exists():
            # self.td_reconstruct.fn_stop_thread()
            self.td_reconstruct.on_closing()

        self.caps.close_all_video_windows()
        cv2.destroyAllWindows()


        # self.logger.warning("Cerrando procesos activos activos.")
        # Agrega aquí cualquier lógica adicional que desees ejecutar antes de cerrar la ventana
        time.sleep(0.5)
        self.destroy()


def main():
    app = MainWindow()
    app.mainloop()


# ==========================================================================

if __name__ == '__main__':
    main()
