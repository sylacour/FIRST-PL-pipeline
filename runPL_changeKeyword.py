#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""

import os
import sys
from astropy.io import fits
from glob import glob
from optparse import OptionParser

# Add options
usage = """
    usage:  %prog files.fits *files.fits

    Goal: update the FIRST_PL Data keywords

    example:
    runPL_changeKeyword.py --DATA_CAT=RAW --DATA_TYP=FLAT

    Fits header Keywords:

    DATA_CAT = RAW , PREPROC, REDUCED
    DATA_TYP =  WAVE, FLAT, SCIENCE, PIXELS, SPECTRA

    DATA_CAT gives the level of reduction:
    RAW means raw data from the camera
    PREPROC means the data has been cut and compressed
    REDUCED means that the data has been reduced

    DATA_TYP gives the type of data
    WAVE is Neon source data
    FLAT is data from SuperK
    SCIENCE is the night time observation data
    But REDUCED data can also have special types:
    PIXELS is the pixel map on the detector
    SPECTRA is the extracted spectra

    Update the keywords.
"""

parser = OptionParser(usage)
parser.add_option("-c","--DATA-CAT", action="store",
                  help="DATA-CAT gives the level of reduction")
parser.add_option("-t","--DATA-TYP", action="store", 
                  help="DATA-TYP gives the type of data")

(argoptions, args) = parser.parse_args()


filelist=[]
## If the user specifies a file name or wild cards ("*_0001.fits")
if len(args) > 0 :
    for f in args:
        filelist += glob(f)
## Processing of the full current directory
else :
    for file in os.listdir("."):
        if file.endswith(".fits"):
            filelist.append(file)

filelist.sort() # process the files in alphabetical order

    
if (argoptions.DATA_TYP!=None)|(argoptions.DATA_CAT!=None):
    for filename in filelist:
        string_print=filename+"   ----->"
        with fits.open(filename, mode='update') as filehandle:
            if argoptions.DATA_CAT:
                filehandle[0].header['DATA-CAT'] = argoptions.DATA_CAT
                string_print+='   DATA-CAT='+argoptions.DATA_CAT
            if argoptions.DATA_TYP:
                filehandle[0].header['DATA-TYP'] = argoptions.DATA_TYP
                string_print+='   DATA-TYP='+argoptions.DATA_TYP
        print(string_print)

