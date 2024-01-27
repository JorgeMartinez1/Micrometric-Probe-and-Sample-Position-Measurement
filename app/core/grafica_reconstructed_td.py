import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import numpy as np
import tkinter as tk
from tkinter import Tk

class GraphTDTk():
    def __init__(self):
        super().__init__()
        self.data_file = f'{os.getcwd()}{os.sep}outputs{os.sep}tdparams{os.sep}puntos3DPatron.txt'
        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        # self.canvas = FigureCanvasTkAgg(self.fig_3d, master=self)
        # self.canvas.get_tk_widget().pack(expand=True, fill="both")

    def algo(self):
        xs, ys, zs = [], [], []

        with open(self.data_file) as file:
            for line in file:
                try:
                    x, y, z = map(float, line.strip().split(','))
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                except Exception as e:
                    print("Error leyendo mediciones", e)

        self.ax_3d.clear()
        self.ax_3d.scatter(xs, ys, zs, c='b')

        self.ax_3d.set_xlabel('x axis')
        self.ax_3d.set_ylabel('y axis')
        self.ax_3d.set_zlabel('z axis')
        self.ax_3d.set_zlim(-2200, 4000)
        self.ax_3d.set_title('3D Graph')

    def animate_3d_puntos_fijos(self, frame):
        self.algo()

    def execute(self, *args):
        self.ani = animation.FuncAnimation(self.fig_3d, self.animate_3d_puntos_fijos, interval=100)
        plt.show(block=False)

# Crear una instancia y ejecutarla
# graph_3d = GraphTDTk()
# graph_3d.execute()



