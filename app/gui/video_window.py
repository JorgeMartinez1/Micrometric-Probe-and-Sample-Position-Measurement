import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk


class VideoWindow(tk.Toplevel):
    def __init__(self, title = ''):
        super().__init__()
        self.title(title)
        self.geometry("640x480")

        self.video_label = tk.Label(self)
        self.video_label.pack(expand=True, fill="both")

        # self.cap = cv2.VideoCapture(index_cam)
        self.cap = cv2.VideoCapture
        self.running = False
        self.protocol("WM_DELETE_WINDOW", self.close_event)

    def execute_video(self, cap:cv2.VideoCapture):
        self.cap = cap
        self.running = True
        self.after(30, self.actualizar_video)

    def actualizar_video(self):
        ret, frame = self.cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = frame_rgb.shape
            image = Image.fromarray(frame_rgb)
            image = ImageTk.PhotoImage(image=image)

            self.video_label.config(image=image)
            self.video_label.image = image  # Mantener una referencia para evitar la recolección de basura

        self.after(30, self.actualizar_video)

    def close_event(self):
        respuesta = messagebox.askquestion("Cerrar Ventana", "¿Estás seguro de que deseas cerrar la ventana?")

        if respuesta == 'yes':
            # Realizar acciones adicionales antes de cerrar la ventana
            # self.cap.release()
            self.running = False
            self.destroy()