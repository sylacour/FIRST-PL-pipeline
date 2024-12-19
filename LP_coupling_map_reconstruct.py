#%%
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
                    'slacour':      '/Users/slacour/FIRST/PhotonicLantern/GitFIRST-PL/config_files/',
                    'jsarrazin':    '/home/jsarrazin/Bureau/PLDATA/FIRST-PL og/FIRST-PL/config_files/'}

CONFIG_DATA_DIR  = {'slacour':      '/Users/slacour/DATA/LANTERNE/RAW_DATA/',
                    'scexao':       '/mnt/tier0/',
                    'first':        '/home/first/Documents/FIRST-DATA/FIRST_PL/',
                    'jsarrazin':    '/home/jsarrazin/Bureau/PLDATA/InitData/CMAP_content/'}

CONFIG_SAVE_DIR  = {'slacour':      '/Users/slacour/DATA/LANTERNE/Test/',
                    'scexao':       '/mnt/userdata/sebviev/FIRST-DATA/Photonic-Lantern/data_red/',
                    'first':        '/home/first/Documents/FIRST-DATA/FIRST_PL/data_red/',
                    'jsarrazin':    '/home/jsarrazin/Bureau/PLDATA/InitData/CMAP_content/'}

PL_COUPLING_ROOT = {'slacour'      :'/home/first/Documents/FIRST-DATA/FIRST_PL/Optim_maps/',
                    'first'      :'/home/first/Documents/FIRST-DATA/FIRST_PL/Optim_maps/',
                    'jsarrazin':    '/home/jsarrazin/Bureau/PLDATA/InitData/CMAP_content/'}

config_filename = CONFIG_FILE_ROOT[getpass.getuser()] + 'config_' + args["<star_name>"] + '_' + args["<date>"] + '.ini'

dr1 = PL_Datareduce(config_filename,CONFIG_DATA_DIR[getpass.getuser()],CONFIG_SAVE_DIR[getpass.getuser()])

dr2 = PL_Datareconstruct(config_filename, PL_COUPLING_ROOT[getpass.getuser()])

# Reconstruct coupling map


print('--------------------------------------------------------------------------')
print('-                   COUPLING MAP RECONSTRUCTION                          -')
print('--------------------------------------------------------------------------')

coupling_cube = dr2.reconstruct_coupling_map(display=True)


print('--------------------------------------------------------------------------')
print('-                 COUPLING MAP PER OUTPUT RECONSTRUCTION                 -')
print('--------------------------------------------------------------------------')

coupling_perout = dr2.reconstruct_coupling_map_per_output(dr1, display=True)


# %%
