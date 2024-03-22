#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase para análisis estadístico de datos

Created on Thus Mar 14 11:09:14 2023
@author: jolumartinez
"""

# Imports
import os
import pathlib
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mpl_toolkits.mplot3d as m3d
import scipy.stats
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# +=================================================================================================



class StatisticalDataAnalisys():
    """Esta clase se crea como recurso para el análisis estadístico rápido de datos. Se espera obtener gráficas
     fundamentales que pueden ser recursos para papers"""

    # Define class attributes
    a_int = 0
    a_list = list()

    # ----------------------------------------------------------------------------------------------
    def __init__(self, arg1=None, arg2=None):
        """Constructor"""

        # Define object attributes
        self.altura = arg1
        self.lista = list()

        # contructor code here

    # ----------------------------------------------------------------------------------------------
    def mymethod(self, arg1, arg2):
        'A method to do some stuff with the attributes of the class'

        # do some stuff here
        print(arg1, arg2, self.a_int)

        return

    def carga_datos_paperIOP(self):
        'A method to do some stuff with the attributes of the class'

        path_base = os.getcwd()
        path_archivos = path_base + os.sep + 'datapaper'
        list_archivos = os.listdir(path_archivos)
        dict_data = {}

        for archivo in list_archivos:
            extension = pathlib.Path(path_archivos + os.sep + archivo).suffix
            if extension == '.csv':
                data = pd.read_csv(path_archivos + os.sep + archivo, header=None)
                axis_me, step_size, axis_size, direction_year, _, _, _, _, _ = archivo.split('-')
                dict_data[archivo] = {'axis': axis_me, 'step_size': step_size, 'axis_size': axis_size,
                                      'direction': direction_year[0], 'X': data[0].tolist(), 'Y': data[1].tolist(),
                                      'Z': data[2].tolist()}

        return dict_data

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def mifuncionestatica(arg1):
        'An example of a function that does not require the object'

        # do some stuff here
        print(arg1)

        return arg1+1

    @staticmethod
    def get_real_values(measured_data) -> list():

        if measured_data[0] < np.array(measured_data).mean():
            num_elements = len(measured_data)  # Desired number of elements
            step = 10  # Step size
            stop = num_elements * step  # Calculate the stop value based on step and desired number of elements

            x = []
            for i in range(num_elements):
                if i == 0:
                    x.append(measured_data[0])
                else:
                    x.append((x[i-1] + 10))

            # print(f"Elements desc Lenx: {str(len(x))}; LenY: {str(len(measured_data))}")
        else:
            num_elements = len(measured_data)  # Desired number of elements

            x = []
            for i in range(num_elements):
                if i == 0:
                    x.append(measured_data[0])
                else:
                    x.append((x[i - 1] - 10))
            # print(f"Elements desc Lenx: {str(len(x))}; LenY: {str(len(measured_data))}")

        return x


    @staticmethod
    def plot_points_and_lineregress(x, y, intercept, slope, line, axis_med:str='X'):

        plt.rcParams.update({'font.size': 14})

        fig, ax = plt.subplots(figsize=(9, 7))

        print(f"Media de y: {str(np.array(y).mean())}")

        # ax.plot(x, y, linewidth=1, marker=',', label=f'')
        ax.plot(x, intercept + slope * np.array(y), label=f'Measured points {axis_med}-Axis (um)')
        ax.plot(x, x, linewidth=1, linestyle='--', label=f'Line-fit with error correction (um)')
        ax.set_xlabel('Distance Moved (um)')
        ax.set_ylabel('Distance Measured (um)')
        ax.legend(facecolor='white', loc='upper left')
        ax.set_title(f"Measured points and Line-fit {axis_med}-Axis")
        # ax.text(0.05, 0.9, f"Mean Error: {mean_error:.2f}\nStd. Error: {std_error:.2f}",
                # transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))


        if axis_med == 'X':
            # Añadir la región de zoom
            zoom_coords = [0.85, 0.06, 0.1, 0.1]  # Ajusta según tus necesidades
            axins = zoomed_inset_axes(ax, zoom=10, loc='lower right', bbox_to_anchor=zoom_coords,
                                      bbox_transform=ax.transAxes)
            # axins = zoomed_inset_axes(ax, zoom=5, loc='lower right')  # Puedes ajustar el valor de 'zoom'
            axins.plot(x, intercept + slope * np.array(y), label=f'Measured points {axis_med}-Axis (um)')
            axins.plot(x, x, linewidth=1, linestyle='--', label=f'Line-fit with error correction (um)')

            # Ajustar la región de zoom
            x1, x2, y1, y2 = 6000, 6300, 6000, 6300  # Puedes ajustar estos valores según tu necesidad
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.8")

        elif axis_med == 'Y':
            zoom_coords = [0.85, 0.06, 0.1, 0.1]  # Ajusta según tus necesidades
            axins = zoomed_inset_axes(ax, zoom=10, loc='lower right', bbox_to_anchor=zoom_coords,
                                      bbox_transform=ax.transAxes)
            # axins = zoomed_inset_axes(ax, zoom=5, loc='lower right')  # Puedes ajustar el valor de 'zoom'
            axins.plot(x, intercept + slope * np.array(y), label=f'Measured points {axis_med}-Axis (um)')
            axins.plot(x, x, linewidth=1, linestyle='--', label=f'Line-fit with error correction (um)')

            # Ajustar la región de zoom
            x1, x2, y1, y2 = -9300, -9000, -9300, -9000  # Puedes ajustar estos valores según tu necesidad
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.8")

        elif axis_med == 'Z':
            zoom_coords = [0.85, 0.06, 0.1, 0.1]  # Ajusta según tus necesidades
            axins = zoomed_inset_axes(ax, zoom=6, loc='lower right', bbox_to_anchor=zoom_coords,
                                      bbox_transform=ax.transAxes)
            # axins = zoomed_inset_axes(ax, zoom=5, loc='lower right')  # Puedes ajustar el valor de 'zoom'
            axins.plot(x, intercept + slope * np.array(y), label=f'Measured points {axis_med}-Axis (um)')
            axins.plot(x, x, linewidth=1, linestyle='--', label=f'Line-fit with error correction (um)')

            # Ajustar la región de zoom
            x1, x2, y1, y2 = 1500, 1800, 1500, 1800  # Puedes ajustar estos valores según tu necesidad
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.8")

        return fig
        # plt.show()

    @staticmethod
    def plot_points_and_lineregress_3D(data, linepts):
        fig, ax = plt.subplots()
        ax = m3d.Axes3D(plt.figure())
        ax.scatter3D(*data.T, marker='.', color='y')
        ax.plot3D(*linepts.T, marker='.')
        return ax.figure
        # plt.show()

    @staticmethod
    def plot_hist_errors_stable(errores):
        # Plot histogram of errors
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(errores, bins=100, density=True,
                alpha=0.7)  # Use density=True for probability density instead of frequency
        ax.set_xlabel('Error in um')
        ax.set_ylabel('Density')
        ax.set_title('Histogram of Errors')

        # Add normal distribution curve
        mu, sigma = np.mean(errores), np.std(errores)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

        # Calculate mean and standard deviation
        mean_error = np.mean(errores)
        std_error = np.std(errores)

        # Print mean and standard deviation
        ax.text(0.05, 0.9, f"Mean Error: {mean_error:.2f}\nStd. Error: {std_error:.2f}",
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

        ax.legend()
        return fig

    from scipy.stats import norm
    @staticmethod
    def plot_hist_errors_old(errors, axis_med):
        # Summary statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)

        # Plot histogram of centered errors
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(errors, bins=20, density=True, alpha=0.7, label='Histogram')
        ax.set_xlabel(f'Errors along {axis_med}-Axis (um)')
        ax.set_ylabel('Density')
        ax.set_title('Histogram of Errors')

        # Add normal distribution curve
        x = np.linspace(min_error, max_error, 100)
        y = norm.pdf(x, mean_error, std_error)
        ax.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
        ax.legend(facecolor='white', loc='upper left')

        deviations = [1, 2, 3]  # Define the number of deviations you want to display
        sigma_symbol = u"\u03C3"  # Sigma symbol (unicode)

        for i in deviations:
            lower = mean_error - i * std_error
            upper = mean_error + i * std_error
            y_range = ax.get_ylim()[1]
            text_offset = 0.005
            ax.axvline(lower, color='g', linestyle='--', linewidth=1)
            ax.axvline(upper, color='g', linestyle='--', linewidth=1)

            middle_value = (lower + upper) / 2
            ax.text(lower, y_range * 0.5, f"{i} {sigma_symbol}", color='g',
                    ha='center', va='center')
            ax.text(upper, y_range * 0.5, f"{i} {sigma_symbol}", color='g',
                    ha='center', va='center')

        ax.axvline(lower, color='g', linestyle='--', linewidth=1, label='Standard Deviations')
        ax.legend(facecolor='white', loc='upper left')
        return fig

    @staticmethod
    def plot_hist_errors(errors, axis_med):
        # Summary statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)

        # Histogram
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].hist(errors, bins=20, density=True, alpha=0.7, label='Histogram')
        axs[0].set_xlabel(f'Errors along {axis_med}-Axis (um)')
        axs[0].set_ylabel('Density')
        axs[0].set_title('Histogram of Errors')

        # Add normal distribution curve
        x = np.linspace(min_error, max_error, 100)
        y = norm.pdf(x, mean_error, std_error)
        axs[0].plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
        axs[0].legend(facecolor='white', loc='upper left')

        deviations = [1, 2, 3]  # Define the number of deviations you want to display
        sigma_symbol = u"\u03C3"  # Sigma symbol (unicode)

        for i in deviations:
            lower = mean_error - i * std_error
            upper = mean_error + i * std_error
            y_range = axs[0].get_ylim()[1]
            text_offset = 0.005
            axs[0].axvline(lower, color='g', linestyle='--', linewidth=1)
            axs[0].axvline(upper, color='g', linestyle='--', linewidth=1)

            middle_value = (lower + upper) / 2
            axs[0].text(lower, y_range * 0.5, f"{i} {sigma_symbol}", color='g',
                        ha='center', va='center')
            axs[0].text(upper, y_range * 0.5, f"{i} {sigma_symbol}", color='g',
                        ha='center', va='center')

        axs[0].axvline(lower, color='g', linestyle='--', linewidth=1, label='Standard Deviations')
        axs[0].legend(facecolor='white', loc='upper left')

        # Box plot
        axs[1].boxplot(errors, vert=False)
        axs[1].set_xlabel('Error')
        axs[1].set_title('Box Plot of Errors')

        # Summary statistics text
        summary_text = f"Mean Error: {mean_error:.2f}\nMedian Error: {median_error:.2f}\nStd. Error: {std_error:.2f}"
        axs[1].text(0.05, 0.05, summary_text, transform=axs[1].transAxes,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))

        return fig

        # plt.tight_layout()
        # plt.show()

    @staticmethod
    def straight_correction(measured_points, original_line):
        # Perform linear regression on the measured points
        measured_points = np.array(measured_points)
        measured_x = measured_points
        measured_y = original_line[0] * measured_points + original_line[1]
        measured_slope, measured_intercept = np.polyfit(measured_x, measured_y, 1)

        # Calculate the correction parameters
        slope_correction = original_line[0] / measured_slope
        intercept_correction = original_line[1] - measured_intercept * slope_correction

        # Apply the correction to the measured line
        corrected_slope = measured_slope * slope_correction
        corrected_intercept = measured_intercept * slope_correction + intercept_correction

        return corrected_slope, corrected_intercept

    @staticmethod
    def ecuacion_recta(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    @staticmethod
    def get_val_recta(valores_medidos, step_size):
        if valores_medidos[0] > valores_medidos[-1]:
            lista_valores_base = []
            for i in range(len(valores_medidos)):
                if i == 0:
                    lista_valores_base.append(valores_medidos[0])
                else:
                    lista_valores_base.append(lista_valores_base[i-1] - float(step_size))

        else:
            lista_valores_base = []
            for i in range(len(valores_medidos)):
                if i == 0:
                    lista_valores_base.append(valores_medidos[0])
                else:
                    lista_valores_base.append(lista_valores_base[i - 1] + float(step_size))

        return lista_valores_base

    import numpy as np
    @staticmethod
    def straight_line_fit(X, Y):
        # Perform linear regression on the points
        slope, intercept = np.polyfit(X, Y, 1)

        return slope, intercept

# +=================================================================================================
def main():
    'A main function, basically to test the class or classes'
    """
    path_base = os.getcwd()
    path_archivos = path_base+os.sep+'datapaper'
    list_archivos = os.listdir(path_archivos)
    dict_data = {}

    for archivo in list_archivos:
        extension = pathlib.Path(path_archivos + os.sep + archivo).suffix
        if extension == '.csv':
            datapaper = pd.read_csv(path_archivos + os.sep + archivo)
            axis_me, step_size, axis_size, direction_year, _, _, _, _, _ = archivo.split('-')
            dict_data[archivo] = {'axis': axis_me, 'step_size': step_size, 'axis_size':axis_size,
                                  'direction': direction_year[0], 'X':datapaper[0].tolist(), 'Y': datapaper[1].tolist(),
                                  'Z': datapaper[2].tolist()}
                                  """
    sda = StatisticalDataAnalisys()
    datos = sda.carga_datos_paperIOP()
    pp = PdfPages('informeGraphs.pdf')

    errors_list = []
    labels_boxplot = []

    for key, value in datos.items():
        x_3d = np.array(value['X'])
        y_3d = np.array(value['Y'])
        z_3d = np.array(value['Z'])
        data_3d = np.concatenate((x_3d[:, np.newaxis], y_3d[:, np.newaxis], z_3d[:, np.newaxis]), axis=1)
        datamean = data_3d.mean(axis=0)
        uu, dd, vv = np.linalg.svd(data_3d - datamean)
        # print(datamean)
        linepts = vv[0] * np.mgrid[-4000:4000:2j][:, np.newaxis]
        linepts += datamean
        # plot_3d = sda.plot_points_and_lineregress_3D(datapaper=data_3d, linepts=linepts)
        # pp.savefig(plot_3d)
        for i in 'XYZ':
            # print(len(value[i]), i)
            if i == value['axis']:
                real_values = sda.get_real_values(value[i])
                slope, intercept, r, *__ = scipy.stats.linregress(np.arange(len(value[i])), real_values)

                slope_corr, inter_corr = sda.straight_line_fit(value[i], real_values)

                corrected_slope, corrected_intercept = sda.straight_correction(measured_points=value[i], original_line=[slope, intercept])

                line = f'Regression line {i}: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

                plot = sda.plot_points_and_lineregress(x=real_values, y=value[i], intercept=inter_corr, slope=slope_corr, line='line', axis_med=value['axis'])
                pp.savefig(plot)

                dpi = 300
                plot.savefig(f"./imgs/points_{value['axis']}_{value['direction']}.png", dpi=dpi)

                corrected_values = []

                for val in value[i]:
                    corrected_values.append((val * slope_corr) + inter_corr)


                valores_recta_base = sda.get_val_recta(value[i], value['step_size'])

                errores = [a - b for a, b in zip(value[i], valores_recta_base)]

                # plot_errores = sda.plot_hist_errors(errores, axis_med=value['axis'])
                # pp.savefig(plot_errores)


                errores_corrected = [a - b for a, b in zip(corrected_values, valores_recta_base)]
                plot_errores = sda.plot_hist_errors(errors=errores_corrected, axis_med=value['axis'])
                pp.savefig(plot_errores)
                dpi = 300
                plot_errores.savefig(f"./imgs/dist_box_{value['axis']}_{value['direction']}.png", dpi=dpi)

                plot_errores = sda.plot_hist_errors_old(errors=errores_corrected, axis_med=value['axis'])
                pp.savefig(plot_errores)
                dpi = 300
                plot_errores.savefig(f"./imgs/dist_{value['axis']}_{value['direction']}.png", dpi=dpi)
                errors_list.append((errores_corrected))
                labels_boxplot.append((f"Error Dist.\n{value['axis']}-Axis"))

    errors_list = errors_list[1:] + [errors_list[0]]
    labels_boxplot = labels_boxplot[1:] + [labels_boxplot[0]]
    fig, ax = plt.subplots()
    ax.boxplot(errors_list)

    ax.set_xlabel('Data Sets')
    ax.set_ylabel('Errors (um)')
    ax.set_title('Error distributions XYZ-Axis')
    ax.set_xticklabels(labels_boxplot)

    pp.savefig(fig)

    pp.close()
    dpi = 300
    fig.savefig(f"./imgs/boxplot_XYZ_{value['axis']}_{value['direction']}.png", dpi=dpi)

    # print(f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}')

    return

# +=================================================================================================
if __name__ == "__main__":
    main()