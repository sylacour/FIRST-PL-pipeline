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


def process_files(folder=".", file_patterns=["*.fits"]):
    """
    Processes files based on the given parameters.

    Args:
        pixel_min (int): Minimum pixel value.
        pixel_max (int): Maximum pixel value.
        pixel_wide (int): Pixel width.
        output_channels (int): Number of output channels.
        file_patterns (list): List of file patterns to process (e.g., ["*.fits"]).
    
    Returns:
        list: A list of files to process.
    """
    filelist = []
    if folder.endswith("*fits"):
        folder = folder[:-5]

    # If file patterns are provided, use glob to find matching files
    for pattern in file_patterns:
        filelist += glob(os.path.join(folder, pattern))

    
    # Sort the file list for consistent processing order
    filelist.sort()
    return filelist


def raw_image_clean(filelist):

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
        
        return raw_image, header

def generate_pixelmap(raw_image, pixel_min, pixel_max, output_channels):
    pixel_length=raw_image.shape[1]

    sampling        = np.linspace(pixel_min+5,pixel_max-5,300,dtype=int)
    peaks           = np.zeros([output_channels, sampling.shape[0]])

    threshold_array=np.linspace(0.01,0.1,50) #originally #np.linspace(0.01,0.1,50) 
    peaks_number=output_channels
    solution_found=[]
    for i in (range(sampling.shape[0])):
        sum_image = raw_image[:,sampling[i]-5:sampling[i]+5].sum(axis=1)
        detectedWavePeaks=np.zeros(output_channels)
        found = False
        for t in threshold_array:
            detectedWavePeaks_tmp = peakutils.peak.indexes(sum_image,thres=t, min_dist=5)
            if len(detectedWavePeaks_tmp) == peaks_number:
                detectedWavePeaks = detectedWavePeaks_tmp
                found = True
                break
        solution_found+=[found]
        peaks[:,i]=detectedWavePeaks

    traces_loc= np.ones([pixel_length,output_channels],dtype=int)

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
            inliers = np.abs(residuals) < 4 * std_residuals

            # Remove outliers
            x = x[inliers]
            y = y[inliers]

        # Fit the polynomial to the cleaned data
        poly_coeffs = np.polyfit(x, y, 1)

        traces_loc[:,i] = np.polyval(poly_coeffs, np.arange(pixel_length))+0.5

        x_found += [x]
        y_found += [y]

    return traces_loc, x_found,y_found

def save_fits_and_png(raw_image,traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, folder):
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
    if folder.endswith("*fits"):
        folder = folder[:-5]
    output_dir = os.path.join(folder,"pixelmaps")

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

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


def run_createPixelMap(folder, destination, pixel_min=100, pixel_max=1600, pixel_wide=3, output_channels=38, file_patterns=["*.fits"]):
    filelist = process_files(folder, pixel_min, pixel_max,pixel_wide,output_channels,file_patterns)
    raw_Image, header = raw_image_clean(filelist)
    traces_loc, x_found,y_found = generate_pixelmap(raw_Image, pixel_min, pixel_max, output_channels)
    save_fits_and_png(raw_Image, traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, folder)
    save_fits_and_png(raw_Image,traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, destination)


if __name__ == "__main__":
    parser = OptionParser(usage)


    # Default values
    pixel_min = 100
    pixel_max = 1600
    pixel_wide = 3
    output_channels = 38
    folder = "."  # Default to current directory

    # Add options for these values
    parser.add_option("--pixel_min", type="int", default=pixel_min,
                    help="Minimum pixel value (default: %default)")
    parser.add_option("--pixel_max", type="int", default=pixel_max,
                    help="Maximum pixel value (default: %default)")
    parser.add_option("--pixel_wide", type="int", default=pixel_wide,
                    help="Pixel width (default: %default)")
    parser.add_option("--output_channels", type="int", default=output_channels,
                    help="Number of output channels (default: %default)")
    
    # Parse the options
    (options, args) = parser.parse_args()

    # Pass the parsed options to the function
    pixel_min=options.pixel_min
    pixel_max=options.pixel_max
    pixel_wide=options.pixel_wide
    output_channels=options.output_channels
    file_patterns=args if args else ['*.fits']
    
    filelist=runlib.get_filelist( file_patterns )

    raw_Image, header = raw_image_clean(filelist)
    traces_loc, x_found,y_found = generate_pixelmap(raw_Image, pixel_min, pixel_max, output_channels)
    save_fits_and_png(raw_Image, traces_loc, header, x_found,y_found, pixel_min, pixel_max,pixel_wide,output_channels, folder)
