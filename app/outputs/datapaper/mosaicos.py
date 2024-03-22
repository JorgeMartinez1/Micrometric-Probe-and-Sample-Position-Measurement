import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Obtén tus imágenes (reemplaza con tus propias imágenes)
imagen1 = plt.imread('./imgs/points_X_H.png')
imagen2 = plt.imread('./imgs/dist_X_H.png')
imagen3 = plt.imread('./imgs/points_Y_A.png')
imagen4 = plt.imread('imgs/dist_Y_A.png')
imagen5 = plt.imread('./imgs/points_Y_H.png')
imagen6 = plt.imread('./imgs/dist_Y_H.png')
imagen7 = plt.imread('./imgs/points_Z_A.png')
imagen8 = plt.imread('imgs/dist_Z_A.png')
imagen9 = plt.imread('./imgs/points_Z_H.png')
imagen10 = plt.imread('./imgs/dist_Z_H.png')

# Crea una figura y define la cuadrícula de subfiguras
fig = plt.figure(figsize=(8, 14))
gs = gridspec.GridSpec(5, 2)

# Define el espacio entre columnas
gs.update(wspace=0.02)

# Coloca las imágenes en las subfiguras
axs = []
axs.append(fig.add_subplot(gs[0, 0]))
axs[-1].imshow(imagen1)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[0, 1]))
axs[-1].imshow(imagen2)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[1, 0]))
axs[-1].imshow(imagen3)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[1, 1]))
axs[-1].imshow(imagen4)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[2, 0]))
axs[-1].imshow(imagen5)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[2, 1]))
axs[-1].imshow(imagen6)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[3, 0]))
axs[-1].imshow(imagen7)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[3, 1]))
axs[-1].imshow(imagen8)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[4, 0]))
axs[-1].imshow(imagen9)
axs[-1].axis('off')

axs.append(fig.add_subplot(gs[4, 1]))
axs[-1].imshow(imagen10)
axs[-1].axis('off')

# Ajusta los espacios entre las filas
gs.update(hspace=0.02)

dpi = 300
fig.savefig(f"./imgs/mosaico.png", dpi=dpi)
