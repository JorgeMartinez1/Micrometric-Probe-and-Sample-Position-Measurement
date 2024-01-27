import logging
import serial
import serial.tools.list_ports
import sys
import os.path
import time

class SerialControl():
    def __init__(self, logger: logging = None):
        super().__init__()
        self.logger = logger
        self.ser_xyz = serial.Serial()
        self.port_xyz = ''
        self.tasa_bs_xyz = 115200
        self.ser_xyz_2 = serial.Serial()
        self.ser_device = serial.Serial()

        self.serial_ports_list = []

        self.logger.info("Se inicializa la clase para control de dispositivos seriales")

    def get_serial_ports(self):
        """Este método retorna una lista con los puertos seriales activos y disponibles en el pc"""
        self.logger.debug(f"Obteniendo listado de puertos seriales disponibles.")
        ports = serial.tools.list_ports.comports()
        # Iterar sobre la lista de puertos seriales y mostrar información de cada uno
        self.serial_ports_list.clear()
        list_ports = []

        for port in ports:
            self.logger.info(
                f"Se encontró el puerto serial: - Nombre: {port.name}, Descripción: {port.description}, Puerto: {port.device}")
            # Agregar el nuevo elemento a la lista
            list_ports.append(str(port.device))
        if len(list_ports) == 0:
            self.logger.warning("No se encontraron puertos seriales disponibles")
            return []
        self.serial_ports_list = list_ports
        return list_ports

    def write_serial_xyz(self, message, port):
        self.port_xyz = port
        if self.ser_xyz.isOpen():
            self.logger.info(f"Enviando mensaje '{str(message)}' a plataforma XYZ")

            mensaje = message + '\n'
            self.ser_xyz.write(mensaje.encode('ascii'))
            time.sleep(0.2)
        else:
            if self.port_xyz == '' or self.port_xyz is None:
                self.logger.error("Debe seleccionar un puerto serial válido.")
                return
            else:
                self.open_serial_xyz()

        if self.ser_xyz.isOpen():
            self.logger.info(f"Enviando mensaje '{str(message)}' a plataforma XYZ")

            mensaje = message + '\n'
            self.ser_xyz.write(mensaje.encode('ascii'))
            time.sleep(0.2)
        else:
            self.logger.critical("No fueposible abrir el puerto serial indicado.")


    def open_serial_xyz(self):
        "Método para configurar y abrir un puerto serial"

        self.logger.debug(f"Abriendo puerto serial {self.port_xyz}")
        rtscts = False
        timeout = 5
        try:
            self.ser_xyz = serial.Serial(
                port=self.port_xyz,
                baudrate=self.tasa_bs_xyz,
                parity=serial.PARITY_NONE,
                rtscts=rtscts,
                timeout=timeout,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
        except Exception as e:
            self.logger.critical(
                f'No se puede establecer comunicación con el puerto serial {str(self.port_xyz)}. Error: {str(e)}')
        time.sleep(0.2)
        self.logger.info(f"Puerto serial {self.port_xyz} abierto")

    def close_all(self):
        if self.ser_xyz.isOpen():
            self.ser_xyz.close()