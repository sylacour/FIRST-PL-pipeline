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

plt.ion()

# Add options
usage = """
    usage:  %prog [options] [directory | files.fits]

    Goal: Preprocess the data using the pixel map.

    Output: files of type DPR_CATG=PREPROC in the preproc directory.
    Also, a figure of the pixel is saved in the preproc directory.
    Also, a figure of the centroid of the data in the pixel map as a function of time.
    This last figure is useful to check if the position of the pixels changed.
    This information (pixel shift) is also stored in the header ('PIX_SHIF').

    Example:
    runPL_preprocess.py --pixel_map=/path/to/pixel_map.fits /path/to/directory

    Options:
    --pixel_map: Force to select which pixel map file to use (default: the one in the directory)
"""




def get_filelist(folder, pixel_map_file):
    filelist = []
    if folder.endswith("*fits"):
        folder = folder[:-5]
    for file in os.listdir(folder):
        if file.endswith(".fits"):
            filelist.append(os.path.join(folder, file))

    # Keys to keep only the RAW files
    fits_keywords = {'DATA-CAT': ['RAW']}
        
    # Use the function to clean the filelist
    filelist_rawdata = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL filelist : ", filelist_rawdata)

    # raise an error if filelist_cleaned is empty
    if len(filelist_rawdata) == 0:
        raise ValueError("No good file to pre-process")

    fits_keywords = {'DATA-CAT': ['PIXELMAP']}
        
    # Use the function to clean the filelist
    pixelmaps_list = []

    if os.path.isfile(pixel_map_file):
        pixelmaps_list.append(pixel_map_file)
    else:
        if not pixel_map_file.endswith(".fits"):
            for file in os.listdir(pixel_map_file):
                if file.endswith(".fits"):
                    pixelmaps_list.append(os.path.join(pixel_map_file,file))

    pixel_map_file = runlib.latest_file(pixelmaps_list)

    filelist_pixelmap = runlib.clean_filelist(fits_keywords, [pixel_map_file])



    # raise an error if filelist_cleaned is empty
    if len(filelist_pixelmap) == 0:
        raise ValueError("No pixel map to pre-process")

    print("Pixel map file ==>> ",filelist_pixelmap)
    # raise an error if filelist_cleaned is empty
    if len(filelist_pixelmap) > 1:
        raise ValueError("Two many pixel maps to use! I can only use one")


    # traces_loc+=5
    # Group files by their directories


    files_by_dir = defaultdict(list)
    for file in filelist_rawdata:
        dir_path = os.path.dirname(os.path.realpath(file))
        files_by_dir[dir_path].append(file)

    return filelist_pixelmap,files_by_dir

def preprocess(filelist_pixelmap,files_by_dir):

    header = fits.getheader(filelist_pixelmap[-1])

    # Read the pixel map header values if they exist, otherwise use defaults
    pixel_min = header.get('PIX_MIN', 100)
    pixel_max = header.get('PIX_MAX', 1600)
    pixel_wide = header.get('PIX_WIDE', 2)
    output_channels = header.get('OUT_CHAN', 38)
    traces_loc=fits.getdata(filelist_pixelmap[-1])

    #%%
    # Process each directory separately 
    for dir_path, files in files_by_dir.items():
        raw_image = None
        center_image = None
        files_out = []
        
        for file in tqdm(files[:], desc=f"Pre-processing of files in {dir_path}"):
            data = fits.getdata(file)
            header = fits.getheader(file)
            object = header.get('OBJECT', "NONAME")
            date = header.get('DATE', 'NODATE')
            type = header.get('DATA-TYP',None)
            date_preproc = datetime.fromtimestamp(os.path.getctime(file)).strftime('%Y-%m-%dT%H:%M:%S')

            header['GAIN'] = 1

            if date == 'NODATE':
                header['DATE'] = date_preproc
                date = date_preproc

            if len(data.shape) == 2:
                data = data[None]

            if raw_image is None:
                raw_image = np.zeros_like(data.sum(axis=0), dtype=np.double)

            raw_image += data.sum(axis=0)
            Nwave = pixel_max - pixel_min
            window_size = (pixel_wide * 2 + 1)

            Nimages = data.shape[0]

            data_cut_pixels = np.zeros((Nimages, output_channels, Nwave, window_size), dtype='uint16')
            data_dark_pixels = np.zeros((Nimages, output_channels - 1, Nwave), dtype='uint16')
            for x in range(Nwave):
                for i in range(output_channels):
                    for w in range(pixel_wide*2+1):
                        t=traces_loc[x + pixel_min, i]+w-pixel_wide
                        if t<0:
                            t=0
                        if t>=data.shape[1]:
                            t=data.shape[1]-1
                        data_cut_pixels[:,i,x,w] = data[:, t, x + pixel_min]
                    if i > 0:
                        t=(traces_loc[x + pixel_min, i-1]+traces_loc[x + pixel_min, i])//2+w-pixel_wide
                        data_dark_pixels[:,i-1,x] = data[:, t, x + pixel_min]
            

            perc_background=np.percentile(data_dark_pixels.ravel(),[50-34.1,50,50+34.1],axis=0)
            data_mean= np.percentile(np.mean(data_cut_pixels,axis=(1,2)),90,axis=0)
            data_cut = np.sum(data_cut_pixels,axis=-1,dtype='uint32')
            flux_mean = np.mean(data_cut,axis=(0,1,2))-perc_background[1]*(pixel_wide*2+1)

            if center_image is None:
                center_image = data_mean[:,None]
            else:
                center_image = np.concatenate((center_image,data_mean[:,None]),axis=1)

            centered=data_mean.argmax()-pixel_wide

            comp_hdu = fits.PrimaryHDU(data_cut, header=header)

            # Update the header with the values read in the headers above
            comp_hdu.header['PIX_MIN'] = pixel_min
            comp_hdu.header['PIX_MAX'] = pixel_max
            comp_hdu.header['PIX_WIDE'] = pixel_wide
            comp_hdu.header['OUT_CHAN'] = output_channels
            comp_hdu.header['PIXELS'] = filelist_pixelmap[-1]
            comp_hdu.header['QC_SHIFT'] = centered
            comp_hdu.header['QC_BACK'] = perc_background[1]
            comp_hdu.header['QC_BACKR'] = (perc_background[2]-perc_background[0])/2*np.sqrt(2)
            comp_hdu.header['QC_FLUX'] = flux_mean
            comp_hdu.header['DATA-CAT'] = "PREPROC"
            # create a directory named preproc if it does not exist
            preproc_dir_path = os.path.join(dir_path, "preproc")
            if not os.path.exists(preproc_dir_path):
                os.makedirs(preproc_dir_path)
            
            output_filename = runlib.create_output_filename(header)
            files_out += [output_filename]
            comp_hdu.writeto(os.path.join(preproc_dir_path, output_filename), overwrite=True, output_verify='fix', checksum=True)
            

        # copy filelist_pixelmap[-1] to the preproc directory
        shutil.copy(filelist_pixelmap[-1], preproc_dir_path)

        # Generate and save the figure for the directory
        fig,ax = runlib.make_figure_of_trace(raw_image, traces_loc, pixel_wide, pixel_min, pixel_max)
        fig.savefig(os.path.join(preproc_dir_path, f"firstpl_"+date_preproc+"_PREPROC.png"), dpi=300)

        # print("file saved as: " + os.path.join(preproc_dir_path, f"firstpl_PIXELS_{os.path.basename(dir_path)}.png"))

        fig = figure("Vertical offset of the dispersed outputs with respect to extracted windows", clear=True, figsize=(5+len(files_out)*0.1, 6))
        imshow(np.log(center_image), aspect='auto', interpolation='none', extent=(-0.5, - 0.5 + len(center_image[0]), +pixel_wide + 0.5, - pixel_wide - 0.5))
        plt.title(f"{fig.get_label()}")
        plt.plot([-0.5, center_image.shape[1] - 0.5], [0, 0], ':', color='k')
        plt.plot(center_image.argmax(axis=0)-pixel_wide, 'o-', color='r')
        plt.xticks(ticks=np.arange(len(files_out)), labels=files_out, rotation=90)
        plt.ylabel("File number")
        plt.ylabel("Pixel shift")
        plt.tight_layout()
        filename_out = os.path.join(preproc_dir_path, f"firstpl_"+date_preproc+"_PREPROCSHIFT.png")
        fig.savefig(filename_out, dpi=300)
        print("PNG saved as: "+filename_out)


def run_preprocess(folder = ".",pixel_map_file = None):
    # Default values
    if folder.endswith("*fits"):
        folder = folder[:-5]
    if pixel_map_file==None :
        pixel_map_file = folder + "pixelmaps"
    
    filelist_pixelmap,files_by_dir = get_filelist(folder, pixel_map_file)
    preprocess(filelist_pixelmap,files_by_dir)

if __name__ == "__main__":
    debug = False

    parser = OptionParser(usage)
    # Default values
    folder ="."

    # Add options for these values
    parser.add_option("--pixel_map", type="string", default=None,
                    help="Force to select which pixel map file to use (default: the one in the directory)")

    (options, args) = parser.parse_args()
    pixel_map_file = options.pixel_map
    folder = args[0] if args else folder

    filelist_pixelmap,files_by_dir = get_filelist(folder, pixel_map_file)
    preprocess(filelist_pixelmap,files_by_dir)
