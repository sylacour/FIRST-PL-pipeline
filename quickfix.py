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
run_preprocess(folder="/home/jsarrazin/Bureau/PLDATA/InitData/Neon2")