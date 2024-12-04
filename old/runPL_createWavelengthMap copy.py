#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
from optparse import OptionParser
import numpy as np
import peakutils

import getpass
import matplotlib
if "VSCODE_PID" in os.environ:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
     
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from datetime import datetime
from tqdm import tqdm
import runPL_library as runlib
import shutil
from collections import defaultdict
from scipy import linalg
from matplotlib import animation
from itertools import product
from scipy.linalg import pinvfind_closest_dark

plt.ion()

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Create a wavelength map from the provided FITS files.

    It will get as input a list of files with DATA-CAT=PREPROC and DATA-TYP=WAVE keywords. 
    It will also find the corresponding dark files with DATA-CAT=PREPROC and DATA-TYP=DARK keywords.
    It will read the wave files and subtract the median of the dark files from them.
    Then, it will find the highest N peaks in the flux and fit a polynomial to create a wavelength map.
    The value N is the number of wavelength provided in th wave_list

    Example:
    runPL_createWavelengthMap.py --wave_list="[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]" *.fits

    Options:
    --wave_list: Comma-separated list of emission lines (default: [753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4])
"""

parser = OptionParser(usage)

# Default values
wave_list_string_default = "[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"
#wave_list_string_default = "[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"
# Add options for these values
parser.add_option("--wave_list", type="string", default=wave_list_string_default,
                  help="comma-separated list of emmission lines (default: %s)"%wave_list_string_default)

# Parse the options and update the values if provided by the user


if "VSCODE_PID" in os.environ:
    filelist = glob("/home/jsarrazin/Bureau/PLDATA/InitData/preproc/*fits")
    wave_list_string = wave_list_string_default
    filelist.sort()  # process the files in alphabetical order
else:
    (options, args) = parser.parse_args()
    wave_list_string = options.wave_list
    filelist = []
    if len(args) > 0:
        for f in args:
            if os.path.isdir(f):
                filelist += glob(os.path.join(f, "*.fit*"))
            else:
                filelist += glob(f)
    # Processing of the full current directory
    else:
        for file in os.listdir("."):
            if file.endswith(".fits"):
                filelist.append(file)

# print(filelist)
# Keys to keep only the RAW files
fits_keywords = {'DATA-CAT': ['PREPROC'], 
                 'DATA-TYP': ['DARK']}
    
# Use the function to clean the filelist
filelist_dark = runlib.clean_filelist(fits_keywords, filelist)

# Keys to keep only the WAVE files
fits_keywords = {'DATA-CAT': ['PREPROC'], 
                 'DATA-TYP': ['WAVE']}
    
# Use the function to clean the filelist
filelist_wave = runlib.clean_filelist(fits_keywords, filelist)

# raise an error if data is not suitable
if len(filelist_wave) == 0:
    raise ValueError("No Neon file to reduce -- are they preprocessed?")
if len(filelist_wave) > 1:
    raise ValueError("Too many Neon file to reduce -- which one shall I use?")
if len(filelist_dark) == 0:
    raise ValueError("No darks")

data_file = filelist_wave[0]

closest_dark_files = runlib.find_closest_dark(data_file, filelist_dark, filter_by_directory = True)

header=fits.getheader(data_file)
data=np.double(fits.getdata(data_file))
data_dark=fits.getdata(closest_dark_files)
data-=np.median(data_dark,axis=0)



#%%
flux = data.mean(axis=(0,1))
flux [flux<1] =1 

threshold_array=np.linspace(0.01,0.3,100)
wavelength_list = [float(w) for w in wave_list_string.strip('[]').split(',')]
peaks_number=len(wavelength_list)
solution_found=[]

found = False
for t in threshold_array:
    detectedWavePeaks_tmp = peakutils.peak.indexes(flux,thres=t, min_dist=6)

    if len(detectedWavePeaks_tmp) == peaks_number: ####== peaks_number:
        detectedWavePeaks = detectedWavePeaks_tmp
        found = True
        break


fig,axs=plt.subplots(2,num="wavelength position",clear=True,figsize=(13,7))
axs[0].plot(flux)
axs[0].plot(np.arange(len(flux))[detectedWavePeaks],flux[detectedWavePeaks],'o')
axs[0].set_title("Peak detected position")
axs[0].set_xlabel("Pixel number")
axs[0].set_ylabel("Flux (ADU)")
axs[0].set_yscale("log")

if found == False:
    output_filename = runlib.create_output_filename(header)
    fig.savefig(output_filename[:-4]+"png",dpi=300)
    print("PNG saved as: "+output_filename[:-4]+"png")
    raise ValueError("Too many Neon file to reduce -- which one shall I use?")


WavePoly=np.polyfit(detectedWavePeaks,wavelength_list,2)
Wavefit=np.poly1d(WavePoly)
pixels=np.arange(flux.shape[0])
pix_to_wavelength_map=Wavefit(pixels)

axs[1].plot(pixels,pix_to_wavelength_map,label='Polynomial fit (deg={})'.format(2))
axs[1].plot(detectedWavePeaks,wavelength_list,'o',label='Detected peaks')
axs[1].set_title("Wavelength vrs Pixels")
axs[1].set_xlabel("Pixel number")
axs[1].set_ylabel("Wavelength (nm)")
axs[1].legend()
fig.tight_layout()
# %%

# Save fits file with traces_loc inside
hdu = fits.PrimaryHDU(pix_to_wavelength_map)
header['DATA-CAT'] = 'WAVEMAP'
# Add date and time to the header
current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
header['DATE-PRO'] = current_time

# Add input parameters to the header
for i in range(len(wavelength_list)):
    header['WAVE%i'%i] = wavelength_list[i]

# Définir le chemin complet du sous-dossier "output/wave"
output_dir = os.path.join("output", "wave")

# Créer les dossiers "output" et "wave" s'ils n'existent pas déjà
os.makedirs(output_dir, exist_ok=True)


hdu.header.extend(header, strip=True)
hdul = fits.HDUList([hdu])
output_filename = os.path.join(output_dir, runlib.create_output_filename(header), filelist_wave[0])
hdul.writeto(output_filename, overwrite=True)

fig.savefig(output_filename[:-4]+"png",dpi=300)

print("File saved as: "+output_filename)
print("PNG saved as: "+output_filename[:-4]+"png")

# %%
