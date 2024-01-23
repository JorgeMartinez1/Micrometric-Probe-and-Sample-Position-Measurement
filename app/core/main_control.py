import tkinter as tk
from tkinter import ttk, scrolledtext
import queue
import logging
import cv2
import threading
import time

from gui.video_window import VideoWindow
from gui.main_window import MainWindow

from core.td_controls import TDControl


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

        self.fn_listar_camaras_disponibles()

        self.cap_left = None
        self.cap_right = None

        self.td_reconstruct = None

    def on_combobox_cam_left_change(self):
        nuevo_valor = self.combo_cam_left.get()
        print(f"Nuevo valor seleccionado: {nuevo_valor}")

    def fn_listar_camaras_disponibles(self):

        for i in range(20):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Agregar opción al Combobox de Tkinter
                self.combo_cam_right['values'] = [*self.combo_cam_right['values'], str(i)]
                self.combo_cam_left['values'] = [*self.combo_cam_left['values'], str(i)]
                logger.info('Se encontró la cámara ' + str(i))
            else:
                break
            cap.release()

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
            try:
                self.cap_left = cv2.VideoCapture(int(cam_str))
                self.video_window_left = VideoWindow(self.cap_left, 'CAMERA LEFT')
                self.video_window_left.mainloop()
            except Exception as e:
                print('Error leyendo cámara: ', str(e))
                return

    def fn_ver_cam_right(self):
        cam_str = self.combo_cam_right.get()
        logger.debug("Se lanza la cámara derecha con id: " + cam_str)

        if cam_str == 'No se encuentran cámaras' or cam_str == '' or cam_str == 'None':
            logger.warning('No ha seleccionado una cámara')
            return
        else:
            try:
                self.cap_right = cv2.VideoCapture(int(cam_str))
                self.video_window_right = VideoWindow(self.cap_right, 'CAMERA RIGHT')
                self.video_window_right.mainloop()
            except Exception as e:
                logger.error('Error leyendo cámara: ', str(e))
                return

    def fn_gen_3d(self):

        cam_str = self.combo_cam_right.get()

        if cam_str == 'No se encuentran cámaras' or cam_str == '' or cam_str == 'None':
            logger.warning('No ha seleccionado una cámara. Se lanzará la reconstrucción en modo simulación.')
            if self.cap_right is not None:
                self.cap_right.release()
                self.cap_right = None
        else:
            try:
                if self.cap_right is not None:
                    self.cap_right.release()

                self.cap_right = cv2.VideoCapture(int(cam_str))
                logger.info(f"Se configra la cámara derecha en {cam_str}")
            except Exception as e:
                logger.error('Error leyendo cámara: ', str(e))
                self.cap_right = None

        cam_str = self.combo_cam_left.get()

        if cam_str == 'No se encuentran cámaras' or cam_str == '' or cam_str == 'None':
            logger.warning('No ha seleccionado una cámara. Se lanzará la reconstrucción en modo simulación.')
            if self.cap_left is not None:
                self.cap_left.release()
                self.cap_left = None
        else:
            try:
                if self.cap_left is not None:
                    self.cap_left.release()

                self.cap_left = cv2.VideoCapture(int(cam_str))
                logger.info(f"Se configra la cámara izquierda en {cam_str}")
            except Exception as e:
                logger.error('Error leyendo cámara: ', str(e))
                self.cap_left = None

        self.td_reconstruct = TDControl(logger, self.cap_left, self.cap_right)
        self.td_reconstruct.mainloop()

    def fn_pictures_reports(self):
        try:
            # Crear e iniciar la ventana de PicturesWindow
            self.pictures_window = tk.Tk()
            pictures_window_instance = PicturesWindow(self.pictures_window)
            pictures_window_instance.show_window()
        except Exception as e:
            logging.critical(f"Error lanzando analizador gráfico de muestras. Error: {str(e)}")

    def fn_config_vna(self):
        try:
            frame_config_vna = ConfigVNA(self, logging)
            logging.debug("Se lanza la interfaz de configuración y control del VNA-PNA")
            # frame_config_vna.move(400, 200)
            frame_config_vna.show()
        except Exception as e:
            logging.error('Error lanzando control de VNA: ' + str(e))

    def fn_get_sample(self):
        get_sample = GetSample(logger, self.ser, self.cap_left, self.cap_right)
        get_sample.mainloop()
        # logging.debug("Se lanza la interfaz de control de la mesa XYZ")
        # frame_control_xyz.show()
    def fn_open_gen_patterns(self):
        frame_patrones = GenPatterns()
        logging.debug("Se lanza la interfaz para generación de patrones")
        frame_patrones.exec_()

    def on_closing(self):
        # Esta función se ejecutará al cerrar la ventana
        self.ser.close_all()
        if self.cap_left is not None:
            self.cap_left.release()
        if self.cap_right is not None:
            self.cap_right.release()

        if self.td_reconstruct is not None:
            # self.td_reconstruct.fn_stop_thread()
            self.td_reconstruct.on_closing()

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
