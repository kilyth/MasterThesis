import numpy as np
import math
import matplotlib.pyplot as plt
from functions.config import *
from functions.data_import import *

def normalize_array(array):
    min = np.min(array)
    max = np.max(array)
    normalized = (array - min) / (max - min)
    return normalized

def plot_3d_slices(data, orientation = 'axial', vmax = False):
    data = scale_range(data, 255)
    pixel_spacing = np.arange(0.0, IMAGE_DIMENSIONS_3D[0], 1), \
                    np.arange(0.0, IMAGE_DIMENSIONS_3D[1], 1), \
                    np.arange(0.0, IMAGE_DIMENSIONS_3D[2], 1),

    if orientation == 'axial':
        n_slices = min(data.shape[2], 32) # plot maximum 32 slices
    else:
        n_slices = 32
    n_cols = 8
    n_rows = math.ceil(n_slices / n_cols)
    aspect_ratio = 1
    base_size = 2
    fig_size = (n_cols*base_size/aspect_ratio, n_rows*base_size)
    fig = plt.figure(figsize=fig_size)

    if orientation == 'axial':
        indeces = np.linspace(start=0, stop=data.shape[2]-1, num=n_slices)
    else:
        indeces = np.linspace(start=0, stop=data.shape[0]-1, num=n_slices)

    sp = 1
    for index in np.nditer(indeces.astype(int)):
        if orientation == 'coronal':
            image = np.rot90(data[index, :, ], k = 3)
            a = pixel_spacing[0]
            b = pixel_spacing[2]
        elif orientation == 'sagital':
            image = np.rot90(data[:, index, :], k = 3)
            a = pixel_spacing[1]
            b = pixel_spacing[2]
        else:
            image = np.rot90(data[:, :, index], k = 2)
            a = pixel_spacing[0]
            b = pixel_spacing[1]
        ax = fig.add_subplot(n_rows, n_cols, sp)
        if vmax:
            ax.pcolormesh(a, b, image, cmap="gray", vmin = 0, vmax = vmax)
        else:
            ax.pcolormesh(a, b, image, cmap="gray")
        ax.set_aspect('equal', 'box')
        ax.set_axis_off()
        sp += 1

    fig.tight_layout(pad=0.0)

def plot_compare(data_dicom, data_jpg, label_dicom, label_jpg, vmax = False, skip = 0):

    data_jpg = normalize_array(data_jpg)
    data_dicom = normalize_array(data_dicom)
    n_slices_dicom = data_dicom.shape[0]
    n_slices_jpg = data_jpg.shape[0]

    n_cols = 2
    n_rows = max(n_slices_dicom, n_slices_jpg)
    aspect_ratio = 1
    base_size = 4
    fig_size = (n_cols*base_size/aspect_ratio, n_rows*base_size)
    fig = plt.figure(figsize=fig_size)

    sp = 1
    for index in range(n_rows):
        try:
            image_dicom = data_dicom[index, :, :, :]
            y_dicom = label_dicom[index]
        except:
            image_dicom = np.zeros((192, 192, 3))
            y_dicom = ""
        try:
            image_jpg = data_jpg[index - skip, :, :, :]
            y_jpg = label_jpg[index - skip]
        except:
            image_jpg = np.zeros((192, 192, 3))
            y_jpg = ""

        ax = fig.add_subplot(n_rows, n_cols, sp)
        if vmax:
            ax.imshow(image_dicom, vmin = 0, vmax = vmax)
        else:
            ax.imshow(image_dicom)
        ax.text(10, 30, index, fontsize = 50, color = "r")
        ax.text(160, 180, y_dicom, fontsize = 50, color = "w")
        ax.set_aspect('equal', 'box')
        ax.set_axis_off()
        sp += 1

        ax = fig.add_subplot(n_rows, n_cols, sp)
        if vmax:
            ax.imshow(image_jpg, vmin = 0, vmax = vmax)
        else:
            ax.imshow(image_jpg)
        ax.text(160, 180, y_jpg, fontsize = 50, color = "w")
        ax.set_aspect('equal', 'box')
        ax.set_axis_off()
        sp += 1

    fig.tight_layout(pad=0.0)
