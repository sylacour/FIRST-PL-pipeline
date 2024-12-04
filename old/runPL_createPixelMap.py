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

plt.ion()

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Create the pixel map needed to preprocess the data.

    Example:
    runPL_createPixelMap.py --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --output_channels=38 *.fits

    Options:
    --pixel_min: Minimum pixel value (default: 100)
    --pixel_max: Maximum pixel value (default: 1600)
    --pixel_wide: Pixel width (default: 3)
    --output_channels: Number of output channels (default: 38)
"""

parser = OptionParser(usage)

# Default values
pixel_min = 100
pixel_max = 1600
pixel_wide = 3
output_channels = 38

# Add options for these values
parser.add_option("--pixel_min", type="int", default=pixel_min,
                  help="Minimum pixel value (default: %default)")
parser.add_option("--pixel_max", type="int", default=pixel_max,
                  help="Maximum pixel value (default: %default)")
parser.add_option("--pixel_wide", type="int", default=pixel_wide,
                  help="Pixel width (default: %default)")
parser.add_option("--output_channels", type="int", default=output_channels,
                  help="Number of output channels (default: %default)")

# Parse the options and update the values if provided by the user


if "VSCODE_PID" in os.environ:
    filelist = glob("/Users/slacour/DATA/LANTERNE/20240917/raw/*fits")
    filelist.sort()  # process the files in alphabetical order
else:
    (options, args) = parser.parse_args()
    pixel_min = options.pixel_min
    pixel_max = options.pixel_max
    pixel_wide = options.pixel_wide
    output_channels = options.output_channels
    filelist = []

    #DEBUG VALUES :
    # pixel_min = 100
    # pixel_max = 1600
    # pixel_wide = 3
    # output_channels = 38
    # filelist = glob("/home/jsarrazin/Bureau/PLDATA/InitData/Neon3/*fits")
    # filelist.sort()  # process the files in alphabetical order

    (argoptions, args) = parser.parse_args()
    if len(args) > 0:
        for f in args:
            filelist += glob(f)
    # Processing of the full current directory
    else:
        for file in os.listdir("."):
            if file.endswith(".fits"):
                filelist.append(file)

# print(filelist)
# Keys to keep only the RAW files
fits_keywords = {'DATA-CAT': ['RAW']}
    
# Use the function to clean the filelist
filelist_cleaned = runlib.clean_filelist(fits_keywords, filelist)

# raise an error if filelist_cleaned is empty
if len(filelist_cleaned) == 0:
    raise ValueError("No good file to process")

header = fits.getheader(filelist_cleaned[-1])

raw_image = np.zeros((header['NAXIS2'], header['NAXIS1']), dtype=np.double)
for filename in tqdm(filelist_cleaned, desc="Co-adding files"):
    raw_image += fits.getdata(filename).sum(axis=0)

pixel_length=raw_image.shape[1]

sampling        = np.linspace(pixel_min+5,pixel_max-5,300,dtype=int)
peaks           = np.zeros([output_channels, sampling.shape[0]])

threshold_array=np.linspace(0.01,0.1,50)
peaks_number=output_channels
solution_found=[]
for i in (range(sampling.shape[0])):
    sum_image = raw_image[:,sampling[i]-5:sampling[i]+5].sum(axis=1)
    detectedWavePeaks=np.zeros(output_channels)
    found = False
    for t in threshold_array:
        detectedWavePeaks_tmp = peakutils.peak.indexes(sum_image,thres=t, min_dist=6)
        if len(detectedWavePeaks_tmp) == peaks_number:
            detectedWavePeaks = detectedWavePeaks_tmp
            found = True
            break
    solution_found+=[found]
    peaks[:,i]=detectedWavePeaks

    # print(len(detectedWavePeaks))

#%%

traces_loc         = np.ones([pixel_length,output_channels],dtype=int)

x_found=[]
y_found=[]

for i in range(output_channels):
    x = sampling[solution_found]
    y = peaks[i][solution_found]

    for b in range(5):
        poly_coeffs = np.polyfit(x, y, 1)

        # Calculate residuals
        y_fit = np.polyval(poly_coeffs, x)
        residuals = y - y_fit

        # Calculate standard deviation of residuals
        std_residuals = np.std(residuals)

        # Identify inliers (points with residuals within the threshold)
        inliers = np.abs(residuals) < 3 * std_residuals

        # Remove outliers
        x = x[inliers]
        y = y[inliers]

    # Fit the polynomial to the cleaned data
    poly_coeffs = np.polyfit(x, y, 1)

    traces_loc[:,i] = np.polyval(poly_coeffs, np.arange(pixel_length))+0.5

    x_found += [x]
    y_found += [y]
    
# create a directory named reduced if it does not exist
# if not os.path.exists('reduced'):
#     os.makedirs('reduced')

# Save fits file with traces_loc inside
hdu = fits.PrimaryHDU(traces_loc)
header['DATA-CAT'] = 'PIXELMAP'
# Add date and time to the header
current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
header['DATE-PRO'] = current_time
if 'DATE' not in header:
    header['DATE'] = current_time

# Add input parameters to the header
header['PIX_MIN'] = pixel_min
header['PIX_MAX'] = pixel_max
header['PIX_WIDE'] = pixel_wide
header['OUT_CHAN'] = output_channels

# Définir le chemin complet du sous-dossier "output/wave"
output_dir = os.path.join("output", "pixel")

# Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
os.makedirs(output_dir, exist_ok=True)


hdu.header.extend(header, strip=True)
hdul = fits.HDUList([hdu])
filename_out = os.path.join(output_dir, runlib.create_output_filename(header))

hdul.writeto(filename_out, overwrite=True)

fig,ax=runlib.make_figure_of_trace(raw_image,traces_loc,pixel_wide,pixel_min,pixel_max)
for i in range(output_channels):
    ax.plot(x_found[i],y_found[i],'w-',linewidth=0.5)

fig.savefig(filename_out[:-4]+"png",dpi=300)

print("File saved as: "+filename_out)
print("PNG saved as: "+filename_out[:-4]+"png")


# %%

