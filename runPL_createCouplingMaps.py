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
from scipy.signal import correlate
from scipy import linalg

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
from collections import defaultdict
from scipy import linalg
from matplotlib import animation
from itertools import product
from scipy.linalg import pinv
from scipy.optimize import curve_fit
import runPL_library as runlib
import runPL_library_imaging as runlib_i
from scipy.ndimage import zoom
from astropy.io import fits
import shutil

plt.ion()

# Add options
usage = """
    usage:  %prog [options] files.fits

    Goal: Compare different coupling maps and make a movie of the correlation between them. Also, plot the deconvolved images.

    It will get as input a list of files with DPR_CATG=CMAP and DPR_TYPE=PREPROC keywords. 
    On those, it will find which ones have the keyword DPR_OPT=DARK and which ones have nothing for DPR_OPT.
    It will read the files which have nothing in the DPR_OPT keyword, and it will subtract from them the files which have the DARK keyword.
    
    Example:
    runPL_compareCouplingMaps.py --cmap_size=25 *.fits

    Options:
    --cmap_size: Width of cmap size, in pixels (default: 25)
"""

def filter_filelist(filelist):

    # Use the function to clean the filelist
    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['OBJECT'],
                    'NAXIS3': [cmap_size*cmap_size]}
    filelist_cmap = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL cmap filelist : ", filelist_cmap)

    fits_keywords = {'DATA-CAT': ['PREPROC'],
                    'DATA-TYP': ['DARK']}
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)
    print("runPL dark filelist : ", filelist_dark)


    # raise an error if filelist_cleaned is empty
    if len(filelist_cmap) == 0:
        raise ValueError("No good file to run cmap")
    # raise an error if filelist_cleaned is empty
    if len(filelist_dark) == 0:
        raise ValueError("No good dark to substract to cmap files")

    # for each file in filelist_cmap find the closest dark file in filelist_dark with, by priority, first the directory in which the file is, and then by the date in the "DATE" fits keyword, and second, the directory in which the file is

    def find_closest_in_time_dark(cmap_file, dark_files):

        cmap_date = fits.getheader(cmap_file)['DATE']
        
        # find the closest by date
        dark_dates = [(dark, fits.getheader(dark)['DATE']) for dark in dark_files]
        dark_dates.sort(key=lambda x: abs(datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S') - datetime.strptime(cmap_date, '%Y-%m-%dT%H:%M:%S')))
        
        return dark_dates[0][0]  # Return the closest dark file by date

    def find_closest_dark(cmap_file, dark_files):
        cmap_dir = os.path.dirname(cmap_file)
        
        # Filter dark files by the same directory
        same_dir_darks = [dark for dark in dark_files if os.path.dirname(dark) == cmap_dir]
        
        if same_dir_darks:
            return find_closest_in_time_dark(cmap_file, same_dir_darks)  # Return the first match in the same directory    
        else:
            return find_closest_in_time_dark(cmap_file, dark_files) 

    closest_dark_files = {cmap: find_closest_dark(cmap, filelist_dark) for cmap in filelist_cmap}

    return closest_dark_files



def extract_datacub(closest_dark_files,wavelength_smooth):

    datacube=[]
    datacube_var=[]
    Nbin=wavelength_smooth
    for data_file,dark_file  in closest_dark_files.items():
        header_tosave=fits.getheader(data_file)

    file_number=1

    for data_file,dark_file  in closest_dark_files.items():
        header=fits.getheader(data_file)
        data_dark=fits.getdata(dark_file)
        if len(data_dark)==1:
            data_dark=data_dark[0]
            data_dark_std=data_dark[0]*0+12
        else:
            data_dark=data_dark.mean(axis=0)
            data_dark_std=data_dark.std(axis=0)
        data=np.double(fits.getdata(data_file))
        data-=data_dark
        gain=header['GAIN']
        data_var=data_dark_std**2+gain*np.abs(data)

        Npos=cmap_size*cmap_size
        Noutput=data.shape[1]
        Nwave=data.shape[2]

        data=data[:,:,:(Nwave//Nbin)*Nbin]
        data_var=data_var[:,:,:(Nwave//Nbin)*Nbin]

        data=data.reshape((Npos,Noutput,Nwave//Nbin,Nbin)).sum(axis=-1)
        data_var=data_var.reshape((Npos,Noutput,Nwave//Nbin,Nbin)).sum(axis=-1)

        Nwave=data.shape[2]

        datacube+=[data]
        datacube_var+=[data_var]

        header_tosave['FILE'+str(file_number)]=os.path.basename(data_file)
        

    return datacube,datacube_var,header_tosave
    
def dithering_of_image(cmap_size, step_size):
    dither_x=[]
    dither_y=[]
    for i in range(cmap_size):
        for j in range(cmap_size):
            dither_x+=[int((i-cmap_size//2)/step_size)]
            dither_y+=[int((j-cmap_size//2)/step_size)]

    dither_x=np.array(dither_x)
    dither_y=np.array(dither_y)
    return (dither_x,dither_y)


def get_shift_between_image(projdata,Ncube):


    def distance_median(dist):
        dist_mean = dist.mean(axis=1)
        dist -= np.round(dist_mean+0.001).astype(int)[:,None]
        return np.round(np.median(dist,axis=0)).astype(int)

    Nsingular=projdata.shape[0]
    cmap_size=int((np.prod(projdata.shape)/Nsingular/Ncube)**.5)

    projdata = projdata.reshape((Nsingular, Ncube, cmap_size, cmap_size))
    # Perform 2D cross-correlation along the last two dimensions
    cross_correlated_projected_data = np.zeros((Nsingular, Ncube, Ncube, cmap_size, cmap_size))

    for i in tqdm(range(Nsingular)):
        for j in range(Ncube):
            for k in range(Ncube):
                cross_correlated_projected_data[i, j, k] = correlate(projdata[i, j], projdata[i, k], mode='same')


    c=cross_correlated_projected_data
    dist=c.sum(axis=0).reshape((Ncube,Ncube,-1)).argmax(axis=2)
    dist_2d_x,dist_2d_y=np.array(np.unravel_index(dist,(cmap_size,cmap_size)))-cmap_size//2
    dist_2d_x=distance_median(dist_2d_x)
    dist_2d_y=distance_median(dist_2d_y)

    print("shift in x --> ",dist_2d_x)
    print("shift in y --> ",dist_2d_y)

    return dist_2d_x,dist_2d_y,c


def get_projection_matrice(datacube,Nsingular):

    U,singular_values,Vh=linalg.svd(datacube,full_matrices=False)

    Ut=U[:,:Nsingular].T

    projection_matrice = Ut

    return projection_matrice,singular_values

def shift_and_add(data, dist_2d_x, dist_2d_y):
    
    Nsingular=data.shape[0]
    Ncube=dist_2d_x.shape[0]
    cmap_size = int(np.sqrt(np.prod(data.shape) / (Nsingular * Ncube)))
    data=data.reshape((Nsingular,Ncube,cmap_size,cmap_size))

    shifted_data = np.zeros_like(data)
    for i in range(Nsingular):
        for j in range(Ncube):
            x_offset = dist_2d_x[j]
            y_offset = dist_2d_y[j]
            shifted_data[i, j] = np.roll(data[i, j], shift=(x_offset, y_offset), axis=(0, 1))

            # Zero padding where we have no information
            if x_offset > 0:
                shifted_data[i, j, :x_offset, :] = np.nan
            elif x_offset < 0:
                shifted_data[i, j, x_offset:, :] = np.nan
            if y_offset > 0:
                shifted_data[i, j, :, :y_offset] = np.nan
            elif y_offset < 0:
                shifted_data[i, j, :, y_offset:] = np.nan

    projected_model=np.nanmean(shifted_data,axis=1)

    return projected_model,shifted_data

def get_fluxtiptilt_model(projected_model):

    Nsingular=projected_model.shape[0]
    cmap_size=projected_model.shape[1]
    fluxtiptilt_model_vectors=np.zeros((cmap_size,cmap_size,Nsingular,3))
    fluxtiptilt_masque=np.zeros((cmap_size,cmap_size),dtype=bool)
    for i in range(cmap_size-1):
        for j in range(cmap_size-1):
            fluxtiptilt_model_vectors[i,j,:,0]=projected_model[:,i,j]
            fluxtiptilt_model_vectors[i,j,:,1]=projected_model[:,i,j]-projected_model[:,i+1,j]
            fluxtiptilt_model_vectors[i,j,:,2]=projected_model[:,i,j]-projected_model[:,i,j+1]
            if np.isnan(fluxtiptilt_model_vectors[i,j]).sum() != 0:
                fluxtiptilt_model_vectors[i,j]=np.nan
            else:
                fluxtiptilt_masque[i,j]=True

    fluxtiptilt_model_vectors=fluxtiptilt_model_vectors[fluxtiptilt_masque]
    norm = np.linalg.norm(fluxtiptilt_model_vectors[:,:,0], axis=1)
    fluxtiptilt_2_projdata_matrix= fluxtiptilt_model_vectors / norm[:,None,None]

    Nmodel=fluxtiptilt_model_vectors.shape[0]
    projdata_2_tiptilts_matrix=np.zeros((Nmodel,3,Nsingular))
    for i in range(Nmodel):
        projdata_2_tiptilts_matrix[i]=pinv(fluxtiptilt_model_vectors[i])

    return fluxtiptilt_2_projdata_matrix,fluxtiptilt_masque,projdata_2_tiptilts_matrix


def interpolate_model(projected_model,interpolation_factor):

    cmap_size=int(projected_model.shape[1]**.5)
    # Interpolate the last two axes of singular_vector_model by a factor of 10
    # interpolation_factor = 20 # 4 magnitudes = 40, 3 magnitudes = 20
    cmap_size2=(cmap_size-1)*interpolation_factor+1
    # cmap_size2=(cmap_size)*interpolation_factor
    projected_model_interpolated = zoom(projected_model, (1, cmap_size2/cmap_size, cmap_size2/cmap_size), order=3)
    norm = np.linalg.norm(projected_model_interpolated, axis=0, keepdims=True)
    projected_model_interpolated /= norm
    projected_model_mask_interpolated = norm[0] > (norm.max() * 0.1)
    masque = projected_model_mask_interpolated
    image_2_projdata_matrix = projected_model_interpolated[:,masque]

    return image_2_projdata_matrix, masque

def get_chi2_maps(projdata,projdata_2_tiptilts_matrix,fluxtiptilt_2_projdata_matrix):

    fluxtiptilt=np.matmul(projdata_2_tiptilts_matrix,projdata)

    projdata_fit = np.matmul(fluxtiptilt_2_projdata_matrix,fluxtiptilt)

    chi2 = np.sum((projdata - projdata_fit)**2, axis=0)
    chi2_max=np.sum((projdata)**2,axis=0)
    chi2_min=chi2.min(axis=0)

    return chi2_min,chi2_max,chi2



if __name__ == "__main__":
    parser = OptionParser(usage)


    # Default values
    cmap_size = 25
    wavelength_smooth = 20
    interpolation_factor = 10
    make_movie = False
    folder = "."  # Default to current directory

    # Add options for these values
    parser.add_option("--cmap_size", type="int", default=cmap_size,
                    help="step numbers of modulation (default: %default)")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")
    parser.add_option("--interpolation_factor", type="int", default=interpolation_factor,
                    help="Interpolation of data between modulation steps (default: %default)")
    parser.add_option("--make_movie", action="store_true", default=make_movie,
                    help="Create a nice mp4 with all datacubes -- can be long (default: %default)")
    
    if "VSCODE_PID" in os.environ:
        file_patterns = "/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc"
        cmap_size = 25
    else:
        # Parse the options
        (options, args) = parser.parse_args()

        # Pass the parsed options to the function
        cmap_size=options.cmap_size
        wavelength_smooth=options.interpolation_factor
        interpolation_factor=options.pixel_wide
        make_movie=options.make_movie
        file_patterns=args if args else ['*.fits']


    filelist=runlib.get_filelist( file_patterns )
    closest_dark_files = filter_filelist(filelist)

    try:
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc/firstpl_2025-01-14T15:34:08_NONAME.fits')

        for _ in range(7):
            closest_dark_files.pop(next(iter(closest_dark_files)))

    except:
        pass

    try:
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:12_NONAME.fits')
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:13_NONAME.fits')
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:14_NONAME.fits')
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:15_NONAME.fits')
    except:
        pass

    
    datacube,datacube_var,header=extract_datacub(closest_dark_files,wavelength_smooth)

    Ncube=len(datacube)
    Npos=datacube[0].shape[0]
    Noutput=datacube[0].shape[1]
    Nwave=datacube[0].shape[2]
    datacube=np.array(datacube).transpose((3,2,0,1))
    datacube_var=np.array(datacube_var).transpose((3,2,0,1))


    Movie=False
    if Movie:
        runlib_i.create_movie_cross(datacube)

        if False:
            plt.close('all')



    # select data only above a threshold based on flux
    datacube=datacube.reshape((Nwave*Noutput,Ncube*Npos))
    flux_thresold=np.percentile(datacube.mean(axis=0),80)/5
    flux_goodData=datacube.mean(axis=0) > flux_thresold

    # get the Nsingulat highest singular values and the projection vectors into that space 
    Nsingular=19*3
    projection_matrix,s=get_projection_matrice(datacube[:,flux_goodData],Nsingular)

    # project the data into that space defined by the above singular values
    projdata= projection_matrix @ datacube

    # cross correlate the dataset to see if there is a significant offset between the different datasets
    dist_2d_x,dist_2d_y,cross_correlated_projected_data = get_shift_between_image(projdata,Ncube)

    # shift and average all the datacubes, do not includes the bad frames
    projdata[:,~flux_goodData]=np.nan
    projected_model,shifted_projected_data = shift_and_add(projdata, dist_2d_x, dist_2d_y)

    # compute the matrices to go from the projected data to the flux and tip tilt (and inverse)
    fluxtiptilt_2_projdata_matrix,masque,projdata_2_tiptilts_matrix = get_fluxtiptilt_model(projected_model)

    #use datamodel to check if the observations are point like
    # To do so, fits the vector model and check if the chi2 decrease resonably
    chi2_min,chi2_max,chi2=get_chi2_maps(projdata,projdata_2_tiptilts_matrix,fluxtiptilt_2_projdata_matrix)
    chi2_delta=chi2_min/chi2_max

    # put a threshold to reduced chi2 that is above 3 sigma of medium value
    percents=np.nanpercentile(chi2_delta,[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2
    np.percentile(chi2_delta.ravel()[~flux_goodData],80)/2
    chi2_goodData = chi2_delta < chi2_threshold

    #redo most of the work above but with flagged datasets
    projection_matrix,s=get_projection_matrice(datacube[:,chi2_goodData],Nsingular)
    projdata= projection_matrix @ datacube
    dist_2d_x,dist_2d_y,cross_correlated_projected_data = get_shift_between_image(projdata,Ncube)
    projdata[:,~chi2_goodData]=np.nan
    projected_model,shifted_projected_data = shift_and_add(projdata, dist_2d_x, dist_2d_y)
    fluxtiptilt_2_projdata_matrix,masque,projdata_2_tiptilts_matrix = get_fluxtiptilt_model(projected_model)

    chi2_min,chi2_max,chi2=get_chi2_maps(projdata,projdata_2_tiptilts_matrix,fluxtiptilt_2_projdata_matrix)
    chi2_delta=chi2_min/chi2_max
    # Save arrays into a FITS file

    output_filename = "output_data.fits"

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu_1_matrix = fits.ImageHDU(data=fluxtiptilt_2_projdata_matrix, name='FTT2PM')
    hdu_2_matrix = fits.ImageHDU(data=projdata_2_tiptilts_matrix, name='P2FTTM')
    hdu_masque = fits.ImageHDU(data=masque.astype(np.uint8), name='MASQUE')  # Save masque as uint8 to save space

    header['DATA-CAT'] = 'COUPLINGMAP'
    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time
    if 'DATE' not in header:
        header['DATE'] = current_time

    # Add input parameters to the header
    header['CMAPSIZE'] = cmap_size  # Add cmap size
    header['WLSMOOTH'] = wavelength_smooth  # Add wavelength smoothing factor

    # Définir le chemin complet du sous-dossier "output/wave"
    if folder.endswith("*fits"):
        folder = folder[:-5]
    output_dir = os.path.join(folder,"couplingmaps")

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # Créer les dossiers "output" et "pixel" s'ils n'existent pas déjà
    os.makedirs(output_dir, exist_ok=True)

    hdu_primary.header.extend(header, strip=True)

    # Combine all HDUs into an HDUList
    hdul = fits.HDUList([hdu_primary, hdu_1_matrix, hdu_2_matrix, hdu_masque])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    hdul.writeto(output_filename, overwrite=True)
    print(f"Data saved to {output_filename}")

    output_plots = output_filename[:-5]+'.pdf'
    runlib_i.generate_plots(s, chi2_delta, flux_goodData, chi2_goodData, chi2_threshold, cross_correlated_projected_data, shifted_projected_data, fluxtiptilt_2_projdata_matrix, output_dir)

# %%
