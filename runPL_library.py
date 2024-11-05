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
import matplotlib.pyplot as plt
from datetime import datetime
import re

def clean_filelist(fits_keywords, filelist, verbose=False):
    filelist_cleaned = []
    for filename in filelist:
        if verbose:
            print(("Check file: " + filename))
        try:
            first_file = fits.open(filename, memmap=True)
        except:
            continue
        header = first_file[0].header.copy()
        first_file.close()
        del first_file

        # Data files with the correct keywords only
        key_names = list(fits_keywords.keys())
        type_ok = True
        for strname in key_names:
            type_ok *= (strname in header)

        if not type_ok:
            if verbose:
                print("DPR_XXX does not exist in header")
            continue

        keys_ok = True
        for name in key_names:
            keys_ok *= (header[name] in fits_keywords[name])

        if not keys_ok:
            if verbose:
                print("DPR_XXX is not set to correct value")
            continue

        filelist_cleaned.append(filename)
    
    return np.sort(filelist_cleaned)


def make_figure_of_trace(raw_image,traces_loc,pixel_wide,pixel_min,pixel_max):
    output_channels=traces_loc.shape[1]
    fig=plt.figure("Extract fitted traces",clear=True,figsize=(18,10))
    v1,v2=np.percentile(raw_image.ravel(),[1,99])
    plt.imshow(raw_image,aspect="auto",interpolation='none',vmin=v1,vmax=v2)
    plt.colorbar()
    for i in range(output_channels): 
            plt.plot(traces_loc[:,i],'r',linewidth=1)
            plt.plot(traces_loc[:,i]+pixel_wide,'g',linewidth=0.3)
            plt.plot(traces_loc[:,i]-pixel_wide,'g',linewidth=0.3)
    plt.plot(np.ones(2)*pixel_min,[0,raw_image.shape[0]],'w')
    plt.plot(np.ones(2)*pixel_max,[0,raw_image.shape[0]],'w')
    plt.xlim(0, raw_image.shape[1])
    plt.ylim(0, raw_image.shape[0])
    plt.tight_layout()
    plt.xlabel("Wavelength")
    ax = plt.gca()
    return fig, ax


def find_closest_in_time_dark(file, dark_files):
    cmap_date = fits.getheader(file)['DATE']
    
    # find the closest by date
    dark_dates = [(dark, fits.getheader(dark)['DATE']) for dark in dark_files]
    dark_dates.sort(key=lambda x: abs(datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S') - datetime.strptime(cmap_date, '%Y-%m-%dT%H:%M:%S')))
    
    return dark_dates[0][0]  # Return the closest dark file by date

def find_closest_dark(file, dark_files, filter_by_directory = False):

    cmap_dir = os.path.dirname(file)
    dark_files = [dark for dark in dark_files if fits.getheader(dark)['GAIN'] == fits.getheader(file)['GAIN']]
    if len(dark_files) == 0:
        raise ValueError("No dark file available with correct gain to reduce file %s"%file)

    # Filter dark files by the same directory
    same_dir_darks = [dark for dark in dark_files if os.path.dirname(dark) == cmap_dir]
    
    if filter_by_directory:
        if same_dir_darks:
            return find_closest_in_time_dark(file, same_dir_darks)  # Return the first match in the same directory    
        else:
            return find_closest_in_time_dark(file, dark_files) 
    else:
        return find_closest_in_time_dark(file, dark_files) 


def create_output_filename(header):
    date = header.get('DATE', 'NODATE')
    object = header.get('OBJECT', "NONAME")
    type = header.get('DATA-TYP',None)
    cat = header.get('DATA-CAT',None)

    name_extension = object
    special_extension = ["DARK", "SKY", "WAVE", "PIXELMAP", "WAVEMAP", "COUPLINGMAP"]
    if type in special_extension:
        name_extension = type
    if cat in special_extension:
        name_extension = cat

    output_filename = 'firstpl_' + date + '_' + name_extension + '.fits'
    return output_filename

def get_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filename)
    if match:
        # Extract date and time parts
        date_part = match.group(1)
        time_part = match.group(2).replace('-', ':')  # Replace '-' with ':' for time
        return f"{date_part}T{time_part}"
    else:
        return None  # Return None if no match is found

