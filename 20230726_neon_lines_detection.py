#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mlallement
"""

###############################################################################
# Python package imports
###############################################################################
# %%
import numpy as np
import numpy.polynomial.polynomial as poly
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from scipy import ndimage
import peakutils
from scipy import optimize as sco
import matplotlib.ticker as ticker
import math


import matplotlib 
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['image.origin'] = 'lower'

###############################################################################
# Functions
###############################################################################


# %%
def gauss(x, x0, amp, wid):
    return amp * np.exp( -(x - x0)**2/(2*wid**2))

def multi_gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        x0 = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + gauss(x, x0, amp, wid)
    return y
#%%
###############################################################################
# Data directories
###############################################################################
data_date = "20231003"
###############################################################################
# Paths
###############################################################################
data_folder = "/home/mlallement/Documents/Doctorat_2022_2023_Linux/2023_Projects/2023_FIRSTv1_R3000/20231003_SpectralResolution/"
###############################################################################
# Get the file names
###############################################################################
# %%

Filename = f"{data_folder}20231003_Neon_FIRSTv1.fits"
Filename_Dark = f"{data_folder}20231003_Neon_FIRSTv1_Dark.fits"
"/home/jsarrazin/Bureau/test zone/"
Filename = "/home/jsarrazin/Bureau/test zone/PL_Neon.fits"
Filename_Dark = "/home/jsarrazin/Bureau/test zone/PL_Neon_dark.fits"

print(Filename)
print(Filename_Dark)

#%%
##############################################################################
# Load and show dark data
##############################################################################
Medians_Dark = np.zeros((1060,2796))
Medians_Dark = np.zeros((412, 1896))
Dark_header = []

Data_Dark, header = fits.getdata(Filename_Dark, header=True)
exp_time_dark = header["ExpTime"]

Medians_Dark[:,:] = np.mean(Data_Dark, axis=0)
Dark_header.append(header)


print(f"Dark shape {Data_Dark.shape}")
print(f"Mean dark shape {Medians_Dark[:,:].shape}")
print(f"Exp time is {exp_time_dark}s")

plt.figure()
plt.imshow(Medians_Dark[:,:])
plt.title(f"Dark - ExpTime: {exp_time_dark}s")
plt.colorbar()


#%%
##############################################################################
# Load and show SW data
##############################################################################
Data_DarkCorr = np.zeros((1060,2796))
Data_DarkCorr = np.zeros((412, 1896))
PolV_header = []

Data_PolV, header = fits.getdata(Filename, header=True)
exp_time = header["ExpTime"]

Data_DarkCorr[:,:] = np.mean(Data_PolV, axis=0) - Medians_Dark
# Medians_PolV[:,:] = Medians_PolV[:,:]/exp_time

PolV_header.append(header)
print(f"Data exp time: {exp_time}")
print(f"Data shape {Data_PolV.shape}")
print(f"Mean data shape {Data_DarkCorr[:,:].shape}")

plt.figure()
plt.title(f"Neon source + FIRST - ExpTime: {exp_time}s")
plt.imshow(Data_DarkCorr[:,:])
plt.colorbar()
plt.show()#%%

# %%
##############################################################################
# Spectra
#############################################################################
#%%
# %%
neon_sum_sp = Data_DarkCorr[10,:]
plt.figure()
plt.plot(neon_sum_sp)
plt.title('Neon source')
plt.legend(bbox_to_anchor = (1.125,1.025))
plt.xlabel('Pixel')
plt.ylabel('Intensity')
#%%
###############################################################################
# Find peaks
###############################################################################

# %%

peaks_index = peakutils.peak.indexes(neon_sum_sp, thres=0.01, min_dist=10)
print(peaks_index.shape)
# peaks_index = np.delete(peaks_index, 5)  # Unknown wavelength
# peaks_index = np.delete(peaks_index, 4)  # Unknown wavelength
peaks_index = np.delete(peaks_index, 3)  # Unknown wavelength
peaks_index = np.delete(peaks_index, 2)  # Unknown wavelength
peaks_index = np.delete(peaks_index, 1)  # Unknown wavelength
peaks_index = np.delete(peaks_index, 0)  # Unknown wavelength

peaks_index = np.delete(peaks_index, 1)  # Unknown wavelength

plt.figure()
plt.plot(neon_sum_sp)
plt.scatter(peaks_index,neon_sum_sp[peaks_index],marker='x',color='black')
plt.title('Neon source experimental spectrum 24th July 2023')
plt.xlabel('Pixel')
plt.ylabel('Intensity')

###############################################################################
# Fit the peaks
###############################################################################
# %%
# Wavelength to fit
# peaks_tofit_nm = [585.249, 588.189, 594.483, 597.553, 603.000, 607.434, 609.616, 614.306, 616.359, 621.728, 
#                   626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 
#                   692.947, 703.241, 717.394, 724.517, 743.890, 748.9, 753.6, 841.8, 849.5, 865.4, 878]


n_pix = neon_sum_sp.shape[0]
x = np.arange(n_pix)
xx = np.arange(0, n_pix, .1) 
neon_sp_red = neon_sum_sp.copy()

plt.figure(f"Peaks detected")
plt.clf()
plt.plot(neon_sum_sp)
plt.scatter(peaks_index,neon_sum_sp[peaks_index],marker='+',color='black', label="Peaks detected")
plt.xlabel("Sensor pixel")
plt.ylabel("Flux (ADU)")
plt.legend()

#%%
gauss_fit_params = []
guess = np.ones(len(peaks_index)* 3, dtype=np.float32)
guess[::3] = peaks_index
guess[1::3] = neon_sum_sp[peaks_index]
guess[2::3] = 3
#%%
plt.figure(f"Multi-Gauss fit for output")
plt.clf()
plt.plot(neon_sum_sp, color='b', label="Measured spectrum")
plt.scatter(guess[::3],guess[1::3],marker='+',color='black', label="Peaks detected")
plt.xlabel("Pixel")
plt.ylabel("Flux (ADU)")
plt.legend()
#%%
popt, pcov = sco.curve_fit(multi_gauss, x, neon_sum_sp, p0=guess)
gauss_fit_params += [popt]
fit = multi_gauss(xx, *popt)
neon_sp_red -= multi_gauss(x, *popt)
peaks_px = popt[::3]
#%%
# Plot the gaussian fit on the data
plt.figure(f"Multi-Gauss fit for output")
plt.clf()
plt.plot(neon_sum_sp, color='b', label="Measured spectrum")
plt.scatter(guess[::3],guess[1::3],marker='+',color='black', label="Peaks detected")
plt.plot(xx, fit, 'r--', label="Multi-Gaussian fit")
plt.xlabel("Pixel")
plt.ylabel("Flux (ADU)")
plt.legend()
#%%

# peaks_tofit_nm = [585.249, 588.189, 594.483, 597.553, 603.000, 607.434, 609.616, 614.306, 616.359, 621.728, 
#                   626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 
#                   692.947, 703.241, 717.394, 724.517, 743.890, 748.9, 753.6, 841.8, 849.5, 865.4, 878]

peaks_tofit_nm = [607.434, 609.616, 614.306, 616.359, 621.728, 626.649, 630.479, 633.443, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704, 
                  692.947, 703.241, 717.394, 724.517, 743.890, 748.9, 753.6, 841.8]
wavelist = "[748.9, 724.5, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 626.7]"

peaks_tofit_nm =[748.9, 724.5, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 626.7]

peaks_tofit_nm.sort(reverse=True)
n_peak_tofit = len(peaks_tofit_nm)
print(n_peak_tofit)

# a=popt[2::3]>0
# a[9]=False

#%%
#MM
# Retrieve peak indexes
peak_index_fit = np.zeros_like(peaks_tofit_nm)
peak_index_fit= popt[::3]
print(peak_index_fit.shape[0])

#%%
# Polynomial fit of the peaks
poly_deg = 2
poly_params, poly_cov = poly.polyfit(peak_index_fit, 
                            peaks_tofit_nm, 
                            poly_deg, 
                            full=True)
poly_fit = poly.polyval(x, poly_params)


#%%
plt.figure()
plt.clf()
plt.plot(neon_sum_sp, color='b', label="Measured spectrum")
plt.plot(xx,fit, color='k', label="Multi-Gaussian fit")
# red dot lines
for i in np.arange(len(peaks_tofit_nm)):
    plt.plot(peak_index_fit[i]*np.ones([1000]),np.linspace(neon_sum_sp[np.int64(peak_index_fit[i])],20000,1000),'r--')
    plt.text(peak_index_fit[i], 20000,
                str(peaks_tofit_nm[i]) + " nm",
                ha='right', rotation=90, color='r', fontsize=10)
plt.xlabel("Pixel")
plt.ylabel("Flux (ADU)")
plt.title('Spectral calibration Neon source - 26th Jan 2023')
plt.ylim([0,25000])
plt.legend()
#%%



#%%
# Plot of the polynomial fit of the peaks

plt.figure(f"Pixel to wavelength fit")
plt.plot(peak_index_fit, peaks_tofit_nm, marker='+', markersize=10, linestyle='')
plt.plot(x, poly_fit, label='final fit ' + r'$\chi^2 = $' + f"{np.round(poly_cov[0][0], decimals=3)}")
plt.xlabel("Pixel")
plt.ylabel("Wavelength (nm)")
plt.legend() 
# np.save('Pixel_axis.npy', x)    

# %%

FWHM_px=2*((2*np.log(2))**0.5)*(abs(popt[2::3]))
print(FWHM_px)
peaks_px= popt[::3]
peaks_nm= peaks_tofit_nm

FWHM_nm = np.zeros_like(FWHM_px)

for i in range(len(peaks_nm)-1):
    print(FWHM_px[i])
    FWHM_nm[i]= -((peaks_nm[i+1]-peaks_nm[i])/(peaks_px[i+1]-peaks_px[i]))*FWHM_px[i]
    #print(FWHM_nm[i])
    print( (peaks_nm[i+1]-peaks_nm[i])/(peaks_px[i+1]-peaks_px[i]) )
Resolution = peaks_nm / FWHM_nm

peaks_nm = np.array(peaks_nm)

import scienceplots
with plt.style.context(['science']):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(np.delete(peaks_nm,-5),np.delete(Resolution,-5),'*',color='black')
    pparams = dict(xlabel="Wavelength (nm)",ylabel="Resolving power")
    axs.set(**pparams)
    # fig.legend(loc='lower center',bbox_to_anchor=(0.5,-0.04), ncols=3)
    fig.savefig("FIRST_FIZ_ResolvingPower.pdf",dpi=800)
plt.show()



# %%
# np.save('Datared/Wavelength_axis_Output37.npy', poly_fit)    
# np.save('Datared/Wavelength_Chi2_Output37.npy', np.round(poly_cov[0][0], decimals=3))   
# np.save('Datared/Resolution_Output4.npy', Resolution) 
# np.save('Datared/Resolution_peaks_Output4.npy', peaks_nm) 
# %%
x = np.arange(0,Medians_Dark.shape[1],1)
wavelength = poly.polyval(x, poly_params)
np.save('20231003.wavelength.npy',wavelength)

import scienceplots
with plt.style.context(['science']):
    fig, axs = plt.subplots(nrows=1, ncols=1,  figsize=(15//2,10//2))
    im= axs.imshow(Medians_Dark[:,:], cmap='magma')
    pparams = dict(xlabel="Wavelength (nm)",ylabel="Pixel")
    axs.set_xticks(np.arange(0,Medians_Dark.shape[1],1)[::200],np.round(wavelength[::200],decimals=1))
    axs.set_yticks(np.arange(0,Medians_Dark.shape[0],1)[::200][1:],np.arange(0,Medians_Dark.shape[0],1)[::200][1:])
    axs.set(**pparams)
    divider = make_axes_locatable(axs)
    axs.set_xlim([0,1600])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical', label='Intensity (ADU)')
    fig.savefig("FIRST_FIZ_Interfero_Dark_50ms.pdf",dpi=600)
#%%
with plt.style.context(['science']):
    fig, axs = plt.subplots(nrows=1, ncols=1,  figsize=(15//2,10//2))
    im= axs.imshow(Data_DarkCorr[:,:],cmap='magma',vmax=250)
    pparams = dict(xlabel="Pixel",ylabel="Pixel")
    # axs.set_xticks(np.arange(0,Medians_Dark.shape[1],1)[::200],np.round(wavelength[::200],decimals=1))
    axs.set_yticks(np.arange(0,Medians_Dark.shape[0],1)[::200][1:],np.arange(0,Medians_Dark.shape[0],1)[::200][1:])
    axs.set(**pparams)
    # axs.set_xlim([0,1600])
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical', label='Intensity (ADU)')
    fig.savefig("FIRST_FIZ_Interfero_Neon_50ms.pdf",dpi=600)

#%%
import scienceplots
with plt.style.context(['science']):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20//2,10//2))
    axs.plot(neon_sum_sp, color='b',linestyle='-', label="Measured spectrum")
    axs.plot(xx,fit, color='k', label="Multi-Gaussian fit")
    for i in np.arange(len(peaks_tofit_nm)):
        if i==0:
            axs.plot(peak_index_fit[i]*np.ones([1000]),np.linspace(neon_sum_sp[np.int64(peak_index_fit[i])],10000,1000),'r--',label='Spectral lines detected')
        axs.plot(peak_index_fit[i]*np.ones([1000]),np.linspace(neon_sum_sp[np.int64(peak_index_fit[i])],11000,1000),'r--')
        axs.text(peak_index_fit[i]+10, 11250,str(np.round(peaks_tofit_nm[i],decimals=1)) + " nm",ha='right', rotation=90, color='r', fontsize=6)
    axs.set_ylim([0,13000])
    # axs.set_xlim([0,1600])
    pparams = dict(xlabel="Pixel",ylabel="Intensity (ADU)")
    axs.set(**pparams)
    fig.legend(loc='lower center',bbox_to_anchor=(0.5,-0.04), ncols=3)
    fig.savefig("FIRST_FIZ_Interfero_SpectralCal.pdf",dpi=800)

print("check")

# %%
