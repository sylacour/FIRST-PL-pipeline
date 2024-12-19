import os
import sys
from astropy.io import fits
from glob import glob
from optparse import OptionParser
import numpy as np
import peakutils
import runPL_library as runlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

def get_preproc(filepath):

    filelist=[]
    for file in os.listdir(filepath):
        if file.endswith(".fits"):
            filelist.append(os.path.join(filepath, file))
    # Keys to keep only the WAVE files
    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['WAVE']}

    # Use the function to clean the filelist
    filelist_wave = runlib.clean_filelist(fits_keywords, filelist)

        # Keys to keep only the WAVE files
    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['DARK']}

    # Use the function to clean the filelist
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)

    data=np.double(fits.getdata(filelist_wave[0]))
    data_dark=fits.getdata(filelist_dark[0])
    data-=np.median(data_dark,axis=0)

    flux = data
    flux = np.mean(flux,axis=2)


    mode = 0
    plt.figure(figsize=(10, 6))
    for mode in range(0,38):
        first_values = [array[mode] for array in flux]

        # Plot the 500 first values
        
        plt.plot(first_values, linestyle='-', label=mode)
    plt.title('Plot of all modes over time')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.show()


    # Create a 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data in 3D
    for mode in range(38):
        first_values = [array[mode] for array in flux]
        z = [mode] * len(first_values)  # Z-axis represents the mode

        # Plot the 500 values for each mode
        ax.plot(range(len(first_values)), first_values, z, label=f'Mode {mode}')

    # Customize the plot
    ax.set_title('3D Plot of All Modes Over Time')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_zlabel('Mode')
    ax.legend()
    plt.show()

def smooth_lambda():
    return

if __name__ == "__main__":
    get_preproc("/home/jsarrazin/Bureau/PLDATA/InitData/Neon1/preproc")