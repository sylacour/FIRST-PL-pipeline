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
    
def latest_file(filelist):

    if filelist==[]:
        return None  # Return None if no valid files are found
    
    # Find the file with the most recent creation time
    last_created_file = max(filelist, key=os.path.getctime)
    
    return last_created_file

def get_fits_date(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            date_str = header.get('DATE', None)
            
            if date_str:
                # Parse the DATE value
                file_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        print(f"Error reading file {fits_file}: {e}")
    return file_date

def get_latest_date_fits(fits_files):
    """
    Finds the FITS file with the latest 'DATE' value in its header.

    Parameters:
        fits_files (list): A list of paths to FITS files.

    Returns:
        str: The path to the FITS file with the latest 'DATE' value.
        None: If no valid 'DATE' values are found in the files.
    """
    latest_file = None
    latest_date = None
    
    for file in fits_files:
        try:
            with fits.open(file) as hdul:
                header = hdul[0].header
                date_str = header.get('DATE', None)
                
                if date_str:
                    # Parse the DATE value
                    file_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                    
                    # Update the latest file if this file has a newer date
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    return latest_file

def get_all_fits_files(folder):
    """
    Recursively finds all FITS files in a given folder and its subfolders.

    Parameters:
        folder (str): The path to the folder to search.

    Returns:
        list: A list of paths to all FITS files in the folder and its subfolders.
    """
    fits_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.fits'):
                fits_files.append(os.path.join(root, file))
    return fits_files

def update_anything_in_fits(file_path, header, header_value):
    # Update the chosen keyword 
    with fits.open(file_path, mode='update') as hdul:
        hdr = hdul[0].header
        hdr[header] = header_value
        hdul.flush()
        print(f"Updated {header} in {file_path} to {header_value}")

def update_anything_in_multiple_fits(folder_path, header, header_value):
    """
    Updates the chosen value in the header of all .fits files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .fits files.
    """
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .fits extension
        if file_name.lower().endswith(".fits"):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # Open the FITS file
                with fits.open(file_path, mode='update') as hdul:
                    update_anything_in_fits(file_path, header, header_value)
            except Exception as e:
                print(f"Failed to update {file_name}: {e}")