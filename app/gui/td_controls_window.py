import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import serial


class TDControlsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controles 3D")

        # --------------- Encabezado -----------------------

        self.label = tk.Label(self, text="Reconstrucción 3D de la escena", font=("Helvetica", 18, "bold"))
        self.label.grid(row=0, column=0, columnspan=5, pady=(10, 0))

        # Frame para la cámara derecha
        self.frame_cam_right = ttk.Frame(self)
        self.frame_cam_right.grid(row=2, column=0, padx=10, pady=10)

        self.label_cam_right = ttk.Label(self.frame_cam_right, text="Cámara Derecha:")
        self.label_cam_right.grid(row=0, column=0, padx=5, pady=5)

        self.combo_cam_right = ttk.Label(self.frame_cam_right, text="None")
        self.combo_cam_right.grid(row=0, column=1, padx=5, pady=5)

        self.btn_ver_cam_right = ttk.Button(self.frame_cam_right, text="Ver", command=self.fn_ver_cam_right)
        self.btn_ver_cam_right.grid(row=0, column=2, padx=5, pady=5)

        # Frame para la cámara izquierda
        self.frame_cam_left = ttk.Frame(self)
        self.frame_cam_left.grid(row=1, column=0, padx=10, pady=10)

        self.label_cam_left = ttk.Label(self.frame_cam_left, text="Cámara Izquierda:")
        self.label_cam_left.grid(row=0, column=0, padx=5, pady=5)

        self.combo_cam_left = ttk.Label(self.frame_cam_left, text="None")
        self.combo_cam_left.grid(row=0, column=1, padx=5, pady=5)

        self.btn_ver_cam_left = ttk.Button(self.frame_cam_left, text="Ver", command=self.fn_ver_cam_left)
        self.btn_ver_cam_left.grid(row=0, column=2, padx=5, pady=5)

        # Frame para los botones
        self.frame_buttons = ttk.Frame(self)
        self.frame_buttons.grid(row=3, column=0, padx=10, pady=10)

        self.btn_both_cams = ttk.Button(self.frame_buttons, text="Generar 3D", command=self.fn_gen_3d)
        self.btn_both_cams.grid(row=0, column=0, padx=5, pady=5)

        # Configurar la función que se ejecutará al cerrar la ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def fn_ver_cam_right(self):
        print("Sobreescribir el método fn_ver_cam_right")

    def fn_ver_cam_left(self):
        print("Sobreescribir el método fn_ver_cam_left")

    def fn_gen_3d(self):
        print("Sobreescribir el método fn_gen_3d")
    def on_closing(self):
        # Esta función se ejecutará al cerrar la ventana
        print("Sobreescribir el método on_closing")
        # Agrega aquí cualquier lógica adicional que desees ejecutar antes de cerrar la ventana
        self.destroy()


# Crear la aplicación Tkinter y ejecutar el bucle principal
if __name__ == "__main__":
    root = tk.Tk()
    app = TDControlsWindow(root)
    root.mainloop()