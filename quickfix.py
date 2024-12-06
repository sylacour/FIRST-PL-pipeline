import os
import sys
from astropy.io import fits
from glob import glob

from runPL_createPixelMap import run_createPixelMap
from runPL_preprocess import run_preprocess

# path = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon3/"
# file = "20230728_FIRST-PL_Neon_cal_Darks.fits"

# file = os.path.join(path,file)
# # Open the FITS file in update mode
# with fits.open(file, mode='update') as hdul:
#     # Access the primary header (or the header of the desired HDU)
#     header = hdul[0].header
#     # Modify the header
#     header['DATA-CAT'] = "RAW"
#     header['DATA-TYP']='DARK'

#     # Save the changes (happens automatically when closing the `with` block)
#     hdul.flush()  # Explicitly write changes to disk (optional)

#run_createPixelMap(folder="/home/jsarrazin/Bureau/PLDATA/InitData/Neon2")
#run_preprocess(folder="/home/jsarrazin/Bureau/PLDATA/InitData/Neon2")


def update_gain_in_fits(folder_path):
    """
    Updates the GAIN value in the header of all .fits files in a specified folder.

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
                    # Access the primary header
                    hdr = hdul[0].header
                    
                    # Update the GAIN value
                    hdr['GAIN'] = 1
                    
                    # Save the changes
                    hdul.flush()
                    print(f"Updated GAIN in {file_name}")
            except Exception as e:
                print(f"Failed to update {file_name}: {e}")

# Example usage
folder = "/home/jsarrazin/Bureau/PLDATA/InitData"

def update_date_in_fits(folder_path):
    """
    Updates the DATE value in the header of all .fits files based on the date in their file names.

    Parameters:
        folder_path (str): Path to the folder containing .fits files.
    """
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".fits"):
            # Extract the date and time from the file name
            try:
                # File name format: im_cube_2024-08-18_11-01-23_altair.fits
                parts = file_name.split("_")
                date_part = parts[2]  # "2024-08-18"
                time_part = parts[3]  # "11-01-23"
                time_part = time_part.split(".")[0]  # Remove ".fits" or extra extensions
                
                # Format as '2024-08-18T11:01:23'
                formatted_date = f"{date_part}T{time_part.replace('-', ':')}"
                
                # Full file path
                file_path = os.path.join(folder_path, file_name)
                
                # Update the DATE keyword in the header
                with fits.open(file_path, mode='update') as hdul:
                    hdr = hdul[0].header
                    hdr['DATE'] = formatted_date
                    hdul.flush()
                    print(f"Updated DATE in {file_name} to {formatted_date}")
            
            except Exception as e:
                print(f"Failed to update DATE in {file_name}: {e}")



def update_header_value(file, which_header, what_value):
    """
    Updates the GAIN value in the header of all .fits files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing .fits files.
    """
    # Check if the file has a .fits extensio
    try:
        # Open the FITS file
        with fits.open(file, mode='update') as hdul:
            # Access the primary header
            hdr = hdul[0].header
            
            # Update the GAIN value
            hdr[which_header] = what_value
            
            # Save the changes
            hdul.flush()
            print(f"Updated {which_header} to {what_value} in {file}")
    except Exception as e:
        print(f"Failed to update {file}: {e}")

update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:57:11.727002706.fits", "DATA-CAT", "RAW")
update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:59:38.528185978.fits", "DATA-CAT", "RAW")
update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:57:11.727002706.fits", "DATA-TYP", "WAVE")
update_header_value("/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/firstpl_00:59:38.528185978.fits", "DATA-TYP", "DARK")


'''
    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['DARK']}

    fits_keywords = {'DATA-CAT': ['PREPROC'], 
                    'DATA-TYP': ['WAVE']}

        fits_keywords = {'DATA-CAT': ['PREPROC'], 
                        'DATA-TYP': ['OBJECT']}


'''