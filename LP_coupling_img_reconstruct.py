'''
    Make_data_reduction.py

    Main entry point for the FIRST Photonic Lantern reduction pipeline for recontruction

    Mandatory arguments <star_name> and <date> are used to generate the ini filename as:
    config_<star_name>_<date>.ini

    Usage:
        Make_data_reduction.py <star_name> <date> [options]

    Example:
        python -i Make_data_reduction.py Algol 20230831
        ipython -i Make_data_reduction.py -- Algol 20230831

        Mind the "--" when using ipython.

    Options:
        <star_name>     Star name as in the .ini filename
        <date>          YYYYMMDD UTC date
        -h|--help       Print this message
        --no-figures    Display and save set to off [Default: False]
        --cal-only      Stop after P2VM and before coherences

'''
#%%
import getpass
if getpass.getuser() == 'slacour':
    import matplotlib
    matplotlib.use('Qt5Agg')
import time
import numpy as np
from astropy.io import fits
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy import ndimage
from scipy import signal
import time
import cmath
import math
from lmfit import Model
from lmfit import Parameters
import peakutils
from tqdm import tqdm
from scipy.fftpack import fft
from scipy.optimize import curve_fit
import json
from scipy import linalg



from FIRSTPL_DataReduction import PL_Datareduce
from FIRSTPL_DataReconstruction import PL_Datareconstruct

plt.ion()

## --------------------------------------------------------------------------
## Initialize the Data reduction environment
## Argument parsing + .ini filename generation
## --------------------------------------------------------------------------
try:
    import docopt
    args = docopt.docopt(__doc__)
except:
    args={'<star_name>': 'LambdaVir',
          "<date>": '20240218',
          '--no-figures': False,
          '--cal-only': False
          }

arg_do_figure = not args['--no-figures']
arg_cal_only = args['--cal-only']
# arg_cp_compute_only = args['--cp-only']

CONFIG_FILE_ROOT = {'first':        '/home/first/FIRST-PL/config_files/',
                    'sebastien':    '/home/sebastien/Documents/Code/FIRST/config_files/',
                    'scexao':       '/home/scexao/sebviev/FIRST-PL/config_files/',
                    'ehuby':        '/home/ehuby/dev/PycharmProjects/firstv1_datared/',
                    'slacour':      '/Users/slacour/FIRST/PhotonicLantern/GitFIRST-PL/config_files/'}

CONFIG_DATA_DIR  = {'slacour':      '/Users/slacour/DATA/LANTERNE/RAW_DATA/',
                    'scexao':       '/mnt/tier0/',
                    'first':        '/home/first/Documents/FIRST-DATA/FIRST_PL/'}

CONFIG_SAVE_DIR  = {'slacour':      '/Users/slacour/DATA/LANTERNE/Test/',
                    'scexao':       '/mnt/userdata/sebviev/FIRST-DATA/Photonic-Lantern/data_red/',
                    'first':        '/home/first/Documents/FIRST-DATA/FIRST_PL/data_red/'}

PL_COUPLING_ROOT = {'slacour'      :'/home/first/Documents/FIRST-DATA/FIRST_PL/Optim_maps/',
                    'first'      :'/home/first/Documents/FIRST-DATA/FIRST_PL/Optim_maps/'}

config_filename = CONFIG_FILE_ROOT[getpass.getuser()] + 'config_' + args["<star_name>"] + '_' + args["<date>"] + '.ini'

dr1 = PL_Datareduce(config_filename,CONFIG_DATA_DIR[getpass.getuser()],CONFIG_SAVE_DIR[getpass.getuser()])

dr2 = PL_Datareconstruct(config_filename, PL_COUPLING_ROOT[getpass.getuser()])

# Reconstruct coupling map


print('--------------------------------------------------------------------------')
print('-                             MAKE F2IM                                  -')
print('--------------------------------------------------------------------------')

# flux = dr2.make_F2IM(dr1)


print('--------------------------------------------------------------------------')
print('-                          RECONSTRUCT IMAGE                             -')
print('--------------------------------------------------------------------------')


#%%

dr1.flatfield_calibration_fit(display=True)

# dr1.wavelength_calibration(display=True, save_plots = True, redo = True)


filename = 'info_'+dr2.coupling_map_date+'_'+dr2.coupling_map_time+'_'+dr2.coupling_map_target+'.txt'
dataname = 'im_cube_'+dr2.coupling_map_date+'_'+dr2.coupling_map_time+'_'+dr2.coupling_map_target+'.fits'
darkname = 'im_dark_'+dr2.coupling_map_date+'_'+dr2.coupling_map_time+'_'+dr2.coupling_map_target+'.fits'

dr2.coupling_map_path="/Users/slacour/DATA/LANTERNE/20240502/2024-05-02_05-23-31_Target1/"
filename="info_2024-05-02_05-23-31_Target1.txt"
dataname="im_cube_2024-05-02_05-23-31_Target1.fits"
darkname = "im_dark_2024-05-02_05-23-31_Target1.fits"

dr2.coupling_map_path="/home/jsarrazin/Bureau/PLDATA/InitData/CMAP_content/"
filename="info_2024-11-21_13-48-32_science.txt"
dataname="im_cube_2024-11-21_13-48-32_science.fits"
darkname = "im_dark_2024-11-21_13-48-32_science.fits"

raw_data = fits.getdata(dr2.coupling_map_path+dataname)
raw_darks = fits.getdata(dr2.coupling_map_path+darkname)

for ii in range(raw_data.shape[0]):
    raw_data[ii] -= raw_darks


img                     = raw_data[:,:,dr1.min_pixel:dr1.max_pixel]
img                     /= img.mean(axis=(1,0))
img_mean                = img.mean(axis=0)

dr1.make_trace_loc(img_mean, display = True)

flux = dr1.extract_trace_spectra(img)




# # sum on the wavelength direction to get 38 peaks
# sum_out                 = np.sum(img_mean, axis=1)
# # detect the positions of traces
# detectedOutPeaks        = peakutils.peak.indexes(sum_out,thres=0.05, min_dist=1)

# # plot spectra for each polar
# polar_1                 = np.array([0,1,2,4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,32,34]) 
# polar_2                 = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,36,37]) 
# pixels_1                = detectedOutPeaks[polar_1]
# pixels_2                = detectedOutPeaks[polar_2]

# flux                    = img[:,pixels_1]+img[:,pixels_1-1]+img[:,pixels_1+1]+img[:,pixels_1-2]+img[:,pixels_1+2]+img[:,pixels_1-3]+img[:,pixels_1+3]

plt.figure();plt.plot(flux.mean(axis=0).mean(axis=0))


# Read the text file
with open(dr2.coupling_map_path+filename, 'r') as file:
    data_txt = json.load(file)

for key, value in data_txt.items():
    globals()[key] = value

npt             = int(globals()['Number points'])
window          = int(globals()['Windows size (mas)'])
x0              = float(globals()['x init'])
y0              = float(globals()['y init'])
x_opt           = float(globals()['x optimal'])
y_opt           = float(globals()['x optimal'])
indx0           = int(npt//2.)
indy0           = int(npt//2.)
pl_outputs_nb   = 38

window_mm       = window*0.047
step_mm         = window_mm/npt
wl_plot         = np.array([200, 400, 600, 800, 1000])
coord_step      = np.linspace(-step_mm/2,step_mm/2,npt)

# spectra = dr1.extract_trace_spectra(data)

# polar_1 = np.array([0,1,2,4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,32,34]) 
# polar_2 = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,36,37]) 

# flux = []
# for i in range(19):
#     flux.append(spectra[:,polar_1[i],:]+spectra[:,polar_2[i],:])

# flux = np.array(flux)#.reshape([flux.shape[1], 19, flux.shape[2]])
# flux = np.reshape(flux,[flux.shape[1], 19, flux.shape[2]])

##############################################################################
#### Build F2IM matrix (Flux-2-Image Matrix)
##############################################################################

U_T     = []
s_T     = []
Vh_T    = []
M_inv   = []

for i in range(flux.shape[2]):
        M_inv+=[linalg.pinv(flux[:,:,i]).T]
        U,s,Vh=linalg.svd(flux[:,:,i].T,full_matrices=False)
        U_T+=[U]
        s_T+=[s]
        Vh_T+=[Vh]

F2IM       = np.array(M_inv)
U               = np.array(U_T)
s               = np.array(s_T)
Vh              = np.array(Vh_T).reshape((-1,19,npt,npt))




# # spectra         = dr1.read_data(dr1.dwd+dr1.data_date+'/orcam_05:47:00.fits',extract_spectra=True)
# # spectra_dark    = dr1.read_data(dr1.dwd+dr1.data_date+'/orcam_05:58:00.fits',extract_spectra=True)
# dataname = 'im_cube_'+dr2.coupling_map_date+'_'+dr2.coupling_map_time+'_'+dr2.coupling_map_target+'.fits'
# darkname = 'im_dark_'+dr2.coupling_map_date+'_'+dr2.coupling_map_time+'_'+dr2.coupling_map_target+'.fits'
# spectra = dr1.read_data(dr2.coupling_map_path+dataname,extract_spectra=True)
# spectra_dark = dr1.read_data(dr2.coupling_map_path+darkname,extract_spectra=True)
# # spectra_dark    = spectra_dark.mean(axis=0)

# for ii in range(spectra.shape[0]):
#     spectra[ii] -= spectra_dark

# polar_1 = np.array([0,1,2,4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,32,34]) 
# polar_2 = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,36,37]) 

# flux = []
# for i in range(19):
#     flux.append(spectra[:,polar_1[i],:]+spectra[:,polar_2[i],:])

# flux = np.array(flux)
# flux = np.reshape(flux,[flux.shape[1], 19, flux.shape[2]])


# flux = flux[60,:,:]

# Init reconstruction images
recons_img = []

# Loop projecting fluxes on F2IM
for i in range(flux.shape[1]):
    Wavelength_channel = i
    F2IM_channel=F2IM[Wavelength_channel]
    recons_img.append(np.dot(F2IM_channel,flux[455,:,Wavelength_channel]))

recons_img = np.array(recons_img)
recons_img = recons_img.reshape((flux.shape[1],31,31))


recons_img_mean = np.mean(recons_img,axis=0)
plt.figure(1)
plt.clf()
plt.imshow(recons_img_mean, origin='lower')
plt.title('Image reconstruction')





# dr2.reconstruct_image(spectra[0],dr1,display=True)
#%%
