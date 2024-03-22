import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Obtén tus imágenes (reemplaza con tus propias imágenes)
imagen1 = plt.imread('./imgs/points_X_A.png')
imagen2 = plt.imread('./imgs/dist_X_A.png')
imagen3 = plt.imread('./imgs/points_Y_A.png')
imagen4 = plt.imread('imgs/dist_Y_A.png')
imagen5 = plt.imread('./imgs/points_Z_A.png')
imagen6 = plt.imread('imgs/dist_Z_A.png')

# Crea una figura y define la cuadrícula de subfiguras
fig = plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(2, 3)

# Define el espacio entre columnas
gs.update(wspace=0.01)

# Ajusta los espacios entre las filas
gs.update(hspace=0.01)

# Coloca las imágenes en las subfiguras
axs = []
axs.append(fig.add_subplot(gs[0, 0]))
axs[-1].imshow(imagen1)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[1, 0]))
axs[-1].imshow(imagen2)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[0, 1]))
axs[-1].imshow(imagen3)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[1, 1]))
axs[-1].imshow(imagen4)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[0, 2]))
axs[-1].imshow(imagen5)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[1, 2]))
axs[-1].imshow(imagen6)
axs[-1].axis('off')



dpi = 900
fig.savefig(f"./imgs/mosaico2.png", dpi=dpi)
