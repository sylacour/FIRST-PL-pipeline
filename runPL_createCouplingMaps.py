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

def filter_filelist(filelist,cmap_size=25):
    """
    Filters the input file list to separate coupling map files and dark files based on FITS keywords.
    Raises an error if no valid files are found.
    Returns a dictionary mapping coupling map files to their closest dark files.
    """

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
        """
        Finds the closest dark file to a given coupling map file based on the 'DATE' FITS keyword.
        """

        cmap_date = fits.getheader(cmap_file)['DATE']
        
        # find the closest by date
        dark_dates = [(dark, fits.getheader(dark)['DATE']) for dark in dark_files]
        dark_dates.sort(key=lambda x: abs(datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S') - datetime.strptime(cmap_date, '%Y-%m-%dT%H:%M:%S')))
        
        return dark_dates[0][0]  # Return the closest dark file by date

    def find_closest_dark(cmap_file, dark_files):
        """
        Finds the closest dark file to a given coupling map file, prioritizing files in the same directory.
        """

        cmap_dir = os.path.dirname(cmap_file)
        
        # Filter dark files by the same directory
        same_dir_darks = [dark for dark in dark_files if os.path.dirname(dark) == cmap_dir]
        
        if same_dir_darks:
            return find_closest_in_time_dark(cmap_file, same_dir_darks)  # Return the first match in the same directory    
        else:
            return find_closest_in_time_dark(cmap_file, dark_files) 

    closest_dark_files = {cmap: find_closest_dark(cmap, filelist_dark) for cmap in filelist_cmap}

    return closest_dark_files


def dithering_of_image(cmap_size, step_size=1):
    """
    Generates dithering offsets for an image based on the cmap size and step size.
    Returns arrays of x and y offsets.
    """

    dither_x=[]
    dither_y=[]
    for i in range(cmap_size):
        for j in range(cmap_size):
            dither_x+=[int((i-cmap_size//2)/step_size)]
            dither_y+=[int((j-cmap_size//2)/step_size)]

    dither_x=np.array(dither_x)
    dither_y=np.array(dither_y)
    return (dither_x,dither_y)


def get_shift_between_image(projdata):
    """
    Calculates the shift between images in a dataset using 2D cross-correlation.
    Returns the x and y shifts, and the cross-correlated data.
    """

    def distance_median(dist):
        dist_mean = dist.mean(axis=1)
        dist -= np.round(dist_mean+0.001).astype(int)[:,None]
        return np.round(np.median(dist,axis=0)).astype(int)

    Nsingular=projdata.shape[0]
    Ncube=projdata.shape[1]
    Npos=projdata.shape[2]  
    cmap_size=int((Npos)**.5)

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


def get_projection_matrice(datacube,flux_goodData,Nsingular):
    """
    Computes the projection matrix and singular values using Singular Value Decomposition (SVD).
    datacube is a flux_2_data matrix
    
        flux_2_data == projdata_2_data @ s @ flux_2_data
        data_2_projdata is the transpose of projdata_2_data

    Returns the projection matrix data_2_projdata and singular values.
    """

    Nwave=datacube.shape[0]
    Noutput=datacube.shape[1]
    Ncube=datacube.shape[2]
    Npos=datacube.shape[3]
    datacube=datacube.reshape((Nwave*Noutput,Ncube,Npos))

    pos_2_data = datacube[:,flux_goodData]

    U,s,Vh=linalg.svd(pos_2_data,full_matrices=False)

    pos_2_singular = Vh[:Nsingular]*s[:Nsingular,None]
    singular_2_data = U[:,:Nsingular]
    pos_2_singular = singular_2_data.T @ datacube.reshape((Nwave*Noutput,Ncube*Npos))

    singular_values = s
    pos_2_singular = pos_2_singular.reshape((Nsingular,Ncube,Npos))
    singular_2_data = singular_2_data.reshape((Nwave,Noutput,Nsingular))

    return pos_2_singular,singular_values,singular_2_data

def shift_and_add(data, dist_2d_x, dist_2d_y):
    """
    Shifts and averages data cubes based on calculated offsets.
    Applies zero padding for areas with no information.
    Returns the averaged model and shifted data.
    """
    
    Nsingular=data.shape[0]
    Ncube=data.shape[1]
    Npos=data.shape[2]
    cmap_size = int(np.sqrt(Npos))
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

    added_data=np.nanmean(shifted_data,axis=1)

    return added_data,shifted_data

def get_postiptilt_model(projected_model,projdata_2_data):
    """
    Computes the flux and tip-tilt model from the projected data.
    Returns matrices for converting between projected data and flux/tip-tilt, and a mask.
    """

    Nsingular=projected_model.shape[0]
    cmap_size=projected_model.shape[1]
    Nwave=projdata_2_data.shape[0]
    Noutput=projdata_2_data.shape[1]

    postiptilt_model_vectors=np.zeros((cmap_size,cmap_size,Nsingular,3))
    postiptilt_masque=np.zeros((cmap_size,cmap_size),dtype=bool)
    for i in range(cmap_size-1):
        for j in range(cmap_size-1):
            postiptilt_model_vectors[i,j,:,0]=projected_model[:,i,j]
            postiptilt_model_vectors[i,j,:,1]=projected_model[:,i,j]-projected_model[:,i+1,j]
            postiptilt_model_vectors[i,j,:,2]=projected_model[:,i,j]-projected_model[:,i,j+1]
            if np.isnan(postiptilt_model_vectors[i,j]).sum() != 0:
                postiptilt_model_vectors[i,j]=0.0
            else:
                postiptilt_masque[i,j]=True

    postiptilt_model_vectors=postiptilt_model_vectors[postiptilt_masque]
    postiptilt_2_projdata= postiptilt_model_vectors

    Nmodel=len(postiptilt_model_vectors)


    a=projdata_2_data.reshape((Nwave*Noutput,Nsingular))
    b=postiptilt_2_projdata.reshape((Nmodel,Nsingular,3))
    postiptilt_2_data = np.matmul(a,b).reshape((Nmodel,Nwave,Noutput,3))
    flux_norm_wave = postiptilt_2_data[:,:,:,0].sum(axis=(0,2), keepdims=True)[:,:,:,None]
    postiptilt_2_data /= flux_norm_wave

    data_2_postiptilt=np.zeros((Nmodel,Nwave,3,Noutput))
    for w in tqdm(range(Nwave)):
        for i in range(Nmodel):
            data_2_postiptilt[i,w]=pinv(postiptilt_2_data[i,w])


    return postiptilt_2_data,data_2_postiptilt,postiptilt_masque

def get_chi2_maps(datacube,postiptilt_2_data,data_2_postiptilt):
    """
    Calculates chi-squared maps to evaluate the fit of the data to the model.
    Returns the minimum chi-squared, maximum chi-squared, and the chi-squared map.
    """

    print("Computing chi2 of indiviual observations")
    Nwave=datacube.shape[0]
    Noutput=datacube.shape[1]
    Ncube=datacube.shape[2]
    Npos=datacube.shape[3]
    Nmodel = postiptilt_2_data.shape[0]

    b=datacube.reshape(Nwave,Noutput,Ncube*Npos)
    ftt = np.matmul(data_2_postiptilt,b)

    chi2=np.zeros((Nmodel,Ncube*Npos))
    for i in tqdm(range(Nmodel)):
        residual = (b-np.matmul(postiptilt_2_data[i],ftt[i]))**2
        chi2[i]= residual.sum(axis=(0,1))

    arg_model=chi2.argmin(axis=0)
    # best_ftt = np.array([ftt[best_model[n],:,:,n] for n in range(Ncube*Npos)])

    chi2_min=chi2.min(axis=0).reshape((Ncube,Npos))
    chi2_max=chi2.max(axis=0).reshape((Ncube,Npos))
    # best_ftt=best_ftt.reshape((Ncube,Npos,Nwave,3))
    arg_model=arg_model.reshape((Ncube,Npos))

    return chi2_min,chi2_max,arg_model


def get_flux_model(postiptilt_2_data):
    Nmodel=postiptilt_2_data.shape[0]
    Nwave=postiptilt_2_data.shape[1]
    Noutput=postiptilt_2_data.shape[2]

    flux_2_data=postiptilt_2_data[:,:,:,0].transpose((1,2,0))
    data_2_flux=np.zeros((Nwave,Nmodel,Noutput))
    for w in tqdm(range(Nwave)):
        data_2_flux[w]=pinv(flux_2_data[w])

    return flux_2_data,data_2_flux

if __name__ == "__main__":
    parser = OptionParser(usage)


    # Default values
    cmap_size = 25
    wavelength_smooth = 20
    wavelength_bin = 15
    interpolation_factor = 10
    make_movie = False
    Nsingular=19*3
    folder = "."  # Default to current directory

    # Add options for these values
    parser.add_option("--cmap_size", type="int", default=cmap_size,
                    help="step numbers of modulation (default: %default)")
    parser.add_option("--Nsingular", type="int", default=Nsingular,
                      help="Number of singular values to use (default: %default)")
    parser.add_option("--wavelength_smooth", type="int", default=wavelength_smooth,
                    help="smoothing factor for wavelength (default: %default)")
    parser.add_option("--wavelength_bin", type="int", default=wavelength_smooth,
                    help="binning factor for wavelength (default: %default)")
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
        Nsingular=options.Nsingular
        wavelength_smooth=options.interpolation_factor
        interpolation_factor=options.pixel_wide
        make_movie=options.make_movie
        wavelength_bin=options.wavelength_bin
        file_patterns=args if args else ['*.fits']


    filelist=runlib.get_filelist( file_patterns )
    closest_dark_files = filter_filelist(filelist,cmap_size)

    try:
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/November2024/preproc/firstpl_2025-01-14T15:34:08_NONAME.fits')

        for _ in range(7):
            closest_dark_files.pop(next(iter(closest_dark_files)))

        # closest_dark_files.pop(next(reversed(closest_dark_files)))
        # closest_dark_files.pop(next(reversed(closest_dark_files)))
        # closest_dark_files.pop(next(reversed(closest_dark_files)))
    except:
        pass

    try:
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:12_NONAME.fits')
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:13_NONAME.fits')
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:14_NONAME.fits')
        closest_dark_files.pop('/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/firstpl_2025-02-19T11:25:15_NONAME.fits')
    except:
        pass
    
    datacube,datacube_var,header=runlib_i.extract_datacube(closest_dark_files,wavelength_smooth,Nbin=wavelength_bin)

    datacube=np.array(datacube).transpose((3,2,0,1))
    datacube_var=np.array(datacube_var).transpose((3,2,0,1))

    Nwave=datacube.shape[0]
    Noutput=datacube.shape[1]
    Ncube=datacube.shape[2]
    Npos=datacube.shape[3]

    Movie=False
    if Movie:
        runlib_i.create_movie_cross(datacube)

        if False:
            plt.close('all')


    # select data only above a threshold based on flux
    flux_thresold=np.percentile(datacube.mean(axis=(0,1)),80)/5
    flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold

    # get the Nsingulat highest singular values and the projection vectors into that space 
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,flux_goodData,Nsingular)

    # cross correlate the dataset to see if there is a significant offset between the different datasets
    dist_2d_x,dist_2d_y,cross_correlated_projected_data = get_shift_between_image(pos_2_singular)

    # shift and average all the datacubes, do not includes the bad frames
    pos_2_singular[:,~flux_goodData]=np.nan
    pos_2_singular_mean,shifted_pos_2_singular = shift_and_add(pos_2_singular, dist_2d_x, dist_2d_y)

    # compute the matrices to go from the projected data to the flux and tip tilt (and inverse)
    postiptilt_2_data,data_2_postiptilt,postiptilt_masque = get_postiptilt_model(pos_2_singular_mean,singular_2_data)

    #use datamodel to check if the observations are point like
    # To do so, fits the vector model and check if the chi2 decrease resonably
    chi2_min,chi2_max,arg_model=get_chi2_maps(datacube,postiptilt_2_data,data_2_postiptilt)
    chi2_delta=chi2_min/chi2_max
    percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
    chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2
    chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData


    #redo most of the work above but with flagged datasets
    pos_2_singular,singular_values,singular_2_data=get_projection_matrice(datacube,chi2_goodData,Nsingular)
    dist_2d_x,dist_2d_y,cross_correlated_projected_data = get_shift_between_image(pos_2_singular)
    pos_2_singular[:,~chi2_goodData]=np.nan
    pos_2_singular_mean,shifted_pos_2_singular = shift_and_add(pos_2_singular, dist_2d_x, dist_2d_y)
    postiptilt_2_data,data_2_postiptilt,postiptilt_masque = get_postiptilt_model(pos_2_singular_mean,singular_2_data)

    flux_2_data,data_2_flux = get_flux_model(postiptilt_2_data)
    # Save arrays into a FITS file


    output_filename = "output_data.fits"

    # Create a primary HDU with no data, just the header
    hdu_primary = fits.PrimaryHDU()

    # Create HDUs for each array
    hdu_0 = fits.ImageHDU(data=postiptilt_masque.astype(np.uint8), name='MASQUE')  # Save masque as uint8 to save space
    hdu_1 = fits.ImageHDU(data=flux_2_data, name='F2DATA')
    hdu_2 = fits.ImageHDU(data=data_2_flux, name='DATA2F')
    hdu_3 = fits.ImageHDU(data=postiptilt_2_data, name='FTT2DATA')
    hdu_4 = fits.ImageHDU(data=data_2_postiptilt, name='DATA2FTT')

    header['DATA-CAT'] = 'COUPLINGMAP'
    # Add date and time to the header
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    header['DATE-PRO'] = current_time
    if 'DATE' not in header:
        header['DATE'] = current_time

    # Add input parameters to the header
    header['CMAPSIZE'] = cmap_size  # Add cmap size
    header['WLSMOOTH'] = wavelength_smooth  # Add wavelength smoothing factor
    header['WL_BIN'] = wavelength_bin
    header['NSINGUL'] = Nsingular  # Add number of singular values

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
    hdul = fits.HDUList([hdu_primary, hdu_0, hdu_1, hdu_2, hdu_3, hdu_4])

    output_filename = os.path.join(output_dir, runlib.create_output_filename(header))

    # Write to a FITS file
    hdul.writeto(output_filename, overwrite=True)
    print(f"Data saved to {output_filename}")

    output_plots = output_filename[:-5]+'.pdf'
    runlib_i.generate_plots(singular_values, chi2_delta, flux_goodData, chi2_goodData, chi2_threshold, cross_correlated_projected_data, shifted_pos_2_singular, postiptilt_2_data, output_dir)

#%%

header = fits.getheader(output_filename)
cmap_file=fits.open(output_filename)
masque=(cmap_file['MASQUE'].data) ==1
flux_2_data=cmap_file['F2DATA'].data
data_2_flux=cmap_file['DATA2F'].data
postiptilt_2_data=cmap_file['FTT2DATA'].data
data_2_postiptilt=cmap_file['DATA2FTT'].data
cmap_file.close()

wavelength_bin = header['WL_BIN']
cmap_size = header['CMAPSIZE']
Nmodel = postiptilt_2_data.shape[0]

datacube,datacube_var,header=runlib_i.extract_datacube(closest_dark_files,Nbin=wavelength_bin)

datacube=np.array(datacube).transpose((3,2,0,1))
datacube_var=np.array(datacube_var).transpose((3,2,0,1))

Nwave=datacube.shape[0]
Noutput=datacube.shape[1]
Ncube=datacube.shape[2]
Npos=datacube.shape[3]

modul_size = 25
dither_x, dither_y = dithering_of_image(modul_size)

# Convert arg_model values into 2D indices of size cmap_size
chi2_min,chi2_max,arg_model = get_chi2_maps(datacube,postiptilt_2_data,data_2_postiptilt)

flux_thresold=np.percentile(datacube.mean(axis=(0,1)),80)/5
flux_goodData=datacube.mean(axis=(0,1)) > flux_thresold
chi2_delta=chi2_min/chi2_max
percents=np.nanpercentile(chi2_delta[flux_goodData],[16,50,84])
chi2_threshold=percents[1]+(percents[2]-percents[0])*3/2
chi2_goodData = (chi2_delta < chi2_threshold)&flux_goodData

arg_model_masques = np.where(masque.ravel())[0][arg_model]

arg_model_indices = np.unravel_index(arg_model_masques, (cmap_size, cmap_size))
arg_model_indices = np.array(arg_model_indices)
# arg_model_indices[0] -= dither_x
# arg_model_indices[1] -= dither_y

fig,ax=plt.subplots(3,num="Position4",clear=True,sharex=True)
x=np.arange(Npos)
for c in range(Ncube):
    ax[0].plot(x[chi2_goodData[c]],arg_model_indices[0][c,chi2_goodData[c]],'.')
    ax[1].plot(x[chi2_goodData[c]],arg_model_indices[1][c,chi2_goodData[c]],'.')
    ax[2].plot(x[chi2_goodData[c]],chi2_delta[c,chi2_goodData[c]],'.-')

ax[0].plot(dither_x)
ax[1].plot(dither_y)

ax[2].set_yscale('log')


#%%
residual = datacube.copy()
fft_fit = np.zeros((Nwave,3,Ncube,Npos))
for c in range(Ncube):
    for p in range(Npos):
        i = arg_model[c,p]
        fft = np.matmul(data_2_postiptilt[i],datacube[:,:,c,p,None])
        fft_fit[:,:,c,p] = fft[:,:,0]
        residual[:,:,c,p] -= np.matmul(postiptilt_2_data[i],fft)[:,:,0]

datacube_cleaned = datacube.copy()
datacube_cleaned[:,:,~chi2_goodData]=0
residual[:,:,~chi2_goodData]=0

image = np.matmul(data_2_flux, datacube_cleaned.reshape((Nwave,Noutput,Ncube*Npos)))
image = image.reshape((Nwave,Nmodel,Ncube,Npos)).transpose((3,1,2,0))
image_2d= runlib_i.resize_and_shift(image,masque, dither_x, dither_y).sum(axis=0)
images_broad=image_2d.sum(axis=3).transpose((2,0,1))

image_residual = np.matmul(data_2_flux, residual.reshape((Nwave,Noutput,Ncube*Npos)))
image_residual = image_residual.reshape((Nwave,Nmodel,Ncube,Npos)).transpose((3,1,2,0))
residual_2d= runlib_i.resize_and_shift(image_residual,masque, dither_x, dither_y).sum(axis=0)
residual_broad=residual_2d.sum(axis=3).transpose((2,0,1))
# Plot all the images in a single figure

fig, axes = plt.subplots(2, len(images_broad), figsize=(15, 6), squeeze=False)

# Normalize color scale across all images
vmin = 0
vmax = max(images_broad.max(), residual_broad.max())/10

# Plot images_broad in the first row
for i, img in enumerate(images_broad):
    ax = axes[0, i]
    im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_title(f"Image {i+1}")
    ax.axis('off')

# Plot residual_broad in the second row
for i, res in enumerate(residual_broad):
    ax = axes[1, i]
    im = ax.imshow(res, vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_title(f"Residual {i+1}")
    ax.axis('off')

# Add a colorbar
# fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

plt.tight_layout()
plt.show()


# %%
