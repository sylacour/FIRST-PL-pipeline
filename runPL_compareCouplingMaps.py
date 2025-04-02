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
from scipy import linalg
from matplotlib import animation
from itertools import product
from scipy.linalg import pinv

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

parser = OptionParser(usage)

# Default values
cmap_size = 25

# Add options for these values
parser.add_option("--cmap_size", type="int", default=cmap_size,
                  help="width of cmap size, in pixel (default: 25)")

# Parse the options and update the values if provided by the user


if "VSCODE_PID" in os.environ:
    filelist = glob("/Users/slacour/DATA/LANTERNE/Optim_maps/2024-08-1[4-5]/*/preproc/*fits")
    filelist = glob("/Users/slacour/DATA/LANTERNE/Optim_maps/202*/*/preproc/*fits")
    filelist = glob("/Users/slacour/DATA/LANTERNE/Optim_maps/2024-05-02/*/preproc/*fits")
    filelist = glob("/home/jsarrazin/Bureau/PLDATA/InitData/Neon1/preproc/*fits")
    filelist = glob("/Users/slacour/DATA/LANTERNE/Optim_maps/May2024/preproc/*fits")
    cmap_size = 19
    # filelist = glob("/Users/slacour/DATA/LANTERNE/Optim_maps/firstpl/*")
    filelist.sort()  # process the files in alphabetical order
else:
    (options, args) = parser.parse_args()
    cmap_size = options.cmap_size
    filelist = []
    if len(args) > 0:
        for f in args:
            if os.path.isdir(f):
                filelist += glob(os.path.join(f, "*.fit*"))
            else:
                filelist += glob(f)
    # Processing of the full current directory
    else:
        for file in os.listdir("."):
            if file.endswith(".fits"):
                filelist.append(file)


# print(filelist)
# Keys to keep only the RAW files
fits_keywords = {'DATA-CAT': ['PREPROC'],
                 'DATA-TYP': ['DARK']}
    
# Use the function to clean the filelist
print("\n")
print(filelist)
print("\n")
filelist_dark = runlib.clean_filelist(fits_keywords, filelist)

# raise an error if filelist_cleaned is empty
if len(filelist_dark) == 0:
    raise ValueError("No good darks")

# Keys to keep only the RAW files
fits_keywords = {'DATA-CAT': ['PREPROC'],
                 'DATA-TYP': ['OBJECT'],
                 'NAXIS3': [cmap_size*cmap_size]}
    
# Use the function to clean the filelist
filelist_cmap = runlib.clean_filelist(fits_keywords, filelist)

#remove the files in filelist_cmap which are in filelist_dark
filelist_cmap = list(set(filelist_cmap) - set(filelist_dark))

filelist_cmap=np.sort(filelist_cmap)[:]

for data_file in filelist_cmap:
    header=fits.getheader(data_file)
    # print(header['NAXIS1'],header['NAXIS2'],header['NAXIS3'],"\t", data_file.split('/')[-3:],"--->"," PIX_SHIF = ",header['PIX_SHIF'])

# Filter out files where the keyword NAXIS1 is below 600
filelist_cmap = [f for f in filelist_cmap if fits.getheader(f)['NAXIS3'] == cmap_size*cmap_size]

# raise an error if filelist_cleaned is empty
if len(filelist_cmap) == 0:
    raise ValueError("No coupling map to reduce -- are they preprocessed?")

print("Coupling Map files:  -->  ",filelist_cmap)

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


all_flux=[]

for data_file,dark_file  in closest_dark_files.items():
    header=fits.getheader(data_file)
    data_dark=fits.getdata(dark_file)
    data=np.double(fits.getdata(data_file))
    data-=data_dark
    # print(data_file.split('/')[-3:],"--->",data.shape,header['PIX_SHIF'])

    flux = data.reshape((len(data),-1))
    all_flux+=[flux.T]


all_flux=np.array(all_flux)

#%%

all_Minv=[]
all_Ut=[]
Nsingular=76
# Nsingular=19

for flux in all_flux:
    U,s,Vh=linalg.svd(flux,full_matrices=False)
    s_inv=1/s[:Nsingular]
    Ut=U[:,:Nsingular].T
    Minv=np.dot(Vh[:Nsingular].T*s_inv,Ut)
    all_Minv+=[Minv]
    all_Ut+=[Ut]


all_Minv=np.array(all_Minv)
all_Ut=np.array(all_Ut)

mp=[]
images=[]
fit_flux=[]
for Ut in all_Ut:
    mp+=[np.matmul(Ut,all_flux)]
for Minv in all_Minv:
    images+=[np.matmul(Minv,all_flux)]
    fit_flux+=[np.matmul(pinv(Minv),images[-1])]
mp=np.array(mp)
images=np.array(images)
fit_flux=np.array(fit_flux)

residuals=fit_flux-all_flux
residuals_std=np.std(residuals,axis=2)

Npts=np.sqrt(images.shape[-1]).astype(int)
Ncmap=images.shape[0]
images=images.reshape((Ncmap*Ncmap,Npts,Npts,-1))
images/=images.max(axis=(1,2,3))[:,None,None,None]

# %%

Movie=False
if Movie:
    print("Making movie ... ")

    def make_image(images,i):
            return images[:,:,i]

    Image=make_image(images[0],0)

    fig,axs = plt.subplots(Ncmap,Ncmap,num=15,figsize=(11,10),clear=True)
    ims=[ax.imshow(Image,vmax=0.2,vmin=-0.1) for ax in axs.ravel()]
    for ax in axs.ravel():
            ax.set_axis_off()
    fig.tight_layout()


    def init():
        for im in ims:
                im.set_array(make_image(images[0],0))
        return ims

    def animate(i):
        for k,im in enumerate(ims):
            im.set_array(make_image(images[k],i))
        return ims

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Npts*Npts, interval=20, blit=True)

    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save('firtpl_CMAP_MOVIE.mp4', writer=FFwriter)

    if False:
        plt.close('all')
#%%

def deconvolve_images(m1, m2):
    m_second=m2.copy()
    Npts = np.sqrt(m1.shape[-1]).astype(int)
    t0 = np.array([np.correlate(m1[i], m1[i], mode='same') for i in range(len(m1))]).sum(axis=0)
    t0_argmax = t0.argmax()
    t0_max = t0.max()
    image_deconvolved = np.zeros((Npts, Npts))
    p_max = []
    flux_max = []
    for _ in range(Npts**2+1):
        t = np.array([np.correlate(m1[i], m_second[i], mode='same') for i in range(len(m1))]).sum(axis=0)
        p_max.append(t.argmax() - t0_argmax)
        flux_max.append(t.max() / t0_max)
        m_second -= np.roll(m1, -p_max[-1]) * flux_max[-1]
        image_deconvolved.ravel()[p_max[-1] + t0_argmax] += flux_max[-1]

    return image_deconvolved


m=np.array([mp[i,i].copy() for i in range(len(mp))])
m_norm=np.sqrt((m**2).sum(axis=2)[:,:,None])
m/=m_norm

image_deconvolved = []
for i in tqdm(range(len(mp))):
    for j in range(len(mp)):
        image_deconvolved.append(deconvolve_images(m[i], mp[i, j] / m_norm[i]))

image_deconvolved = np.array(image_deconvolved)


fig,axs = plt.subplots(Ncmap,Ncmap,num=16,figsize=(11,10),clear=True)

for k,ax in enumerate(axs.ravel()):
    ax.imshow(image_deconvolved[k])
    ax.set_axis_off()
plt.tight_layout()
fig.savefig('firtpl_CMAP_CORRELATION.png', dpi=300)

# %%
