import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Control Pannel Microwave Microscope by Jorge Martínez")
        # self.create_widgets()
        self.pictures_window = None  # Instancia de PicturesWindow

        # Frame para la cámara derecha
        self.frame_cam_right = ttk.Frame(self)
        self.frame_cam_right.grid(row=1, column=0, padx=10, pady=10)

        self.label_cam_right = ttk.Label(self.frame_cam_right, text="Cámara Derecha:")
        self.label_cam_right.grid(row=0, column=0, padx=5, pady=5)

        self.combo_cam_right = ttk.Combobox(self.frame_cam_right, values=["None"])
        self.combo_cam_right.grid(row=0, column=1, padx=5, pady=5)

        self.btn_ver_cam_right = ttk.Button(self.frame_cam_right, text="Ver", command=self.fn_ver_cam_right)
        self.btn_ver_cam_right.grid(row=0, column=2, padx=5, pady=5)

        # Frame para la cámara izquierda
        self.frame_cam_left = ttk.Frame(self)
        self.frame_cam_left.grid(row=0, column=0, padx=10, pady=10)

        self.label_cam_left = ttk.Label(self.frame_cam_left, text="Cámara Izquierda:")
        self.label_cam_left.grid(row=0, column=0, padx=5, pady=5)

        self.combo_cam_left = ttk.Combobox(self.frame_cam_left, values=["None"])
        self.combo_cam_left.grid(row=0, column=1, padx=5, pady=5)
        # Asociar la función al evento <ComboboxSelected> con una lambda
        self.combo_cam_left.bind("<<ComboboxSelected>>", lambda event: self.on_combobox_cam_left_change())

        self.btn_ver_cam_left = ttk.Button(self.frame_cam_left, text="Ver", command=self.fn_ver_cam_left)
        self.btn_ver_cam_left.grid(row=0, column=2, padx=5, pady=5)

        # Frame para los botones
        self.frame_buttons = ttk.Frame(self)
        self.frame_buttons.grid(row=2, column=0, padx=10, pady=10)

        self.btn_both_cams = ttk.Button(self.frame_buttons, text="Ver Ambas Cámaras")
        self.btn_both_cams.grid(row=0, column=0, padx=5, pady=5)

        self.btn_gen_patterns = ttk.Button(self.frame_buttons, text="Generar Patrones")
        self.btn_gen_patterns.grid(row=0, column=1, padx=5, pady=5)

        self.btn_cam_calibration = ttk.Button(self.frame_buttons, text="Calibrar Cámaras")
        self.btn_cam_calibration.grid(row=0, column=2, padx=5, pady=5)

        self.btn_gen_3d = ttk.Button(self.frame_buttons, text="Generar 3D", command=self.fn_gen_3d)
        self.btn_gen_3d.grid(row=0, column=3, padx=5, pady=5)

        self.btn_measur_stimation = ttk.Button(self.frame_buttons, text="Ubicar Punta")
        self.btn_measur_stimation.grid(row=1, column=0, padx=5, pady=5)

        self.btn_xyz_control = ttk.Button(self.frame_buttons, text="Control XYZ", command=self.fn_get_sample)
        self.btn_xyz_control.grid(row=1, column=1, padx=5, pady=5)

        self.btn_vna = ttk.Button(self.frame_buttons, text="Conectar VNA-PNA", command=self.fn_connect_vna_pna)
        self.btn_vna.grid(row=1, column=2, padx=5, pady=5)

        self.btn_figures = ttk.Button(self.frame_buttons, text="Gráficas e Informes", command=self.fn_pictures_reports)
        self.btn_figures.grid(row=1, column=3, padx=5, pady=5)

        self.text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=10, width=60, font=("Times New Roman", 12),
                                                   state='disabled')
        # self.text_area.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.text_area.tag_config('INFO', foreground='green')
        self.text_area.tag_config('DEBUG', foreground='gray')
        self.text_area.tag_config('WARNING', foreground='orange')
        self.text_area.tag_config('ERROR', foreground='red')
        self.text_area.tag_config('CRITICAL', foreground='red', underline=1)
        self.text_area.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.adjust_geometry()

        # Configurar la función que se ejecutará al cerrar la ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_combobox_cam_left_change(self):
        print(f"Sobreescribir el método on_combobox_cam_left_change")

    def adjust_geometry(self):
        # Ajustar la geometría de la ventana al contenido
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")

    def fn_ver_cam_right(self):
        print(f"Sobreescribir el método fn_ver_cam_right")

    def fn_ver_cam_left(self):
        print(f"Sobreescribir el método fn_ver_cam_left")

    def fn_pictures_reports(self):
        print(f"Sobreescribir el método fn_pictures_reports")

    def fn_get_sample(self):
        print(f"Sobreescribir el método fn_get_sample")

    def fn_connect_vna_pna(self):
        print(f"Sobreescribir el método fn_connect_vna_pna")

    def fn_gen_3d(self):
        print(f"Sobreescribir el método fn_gen_3d")

    def on_closing(self):
        # Esta función se ejecutará al cerrar la ventana
        print("Sobreescribir el método on_closing")
        # Agrega aquí cualquier lógica adicional que desees ejecutar antes de cerrar la ventana
        self.destroy()