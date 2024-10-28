# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
# C:\Users\Jehanne\Desktop\Doctorat\FIRST-PL\Neon1\PL_Neon.fits
from astropy.utils.data import download_file
image_file = "C:/Users/Jehanne/Desktop/Doctorat/FIRST-PL/Neon1/PL_Neon.fits"
dark_file = "C:/Users/Jehanne/Desktop/Doctorat/FIRST-PL/Neon1/PL_Neon_dark.fits"

hdu_list = fits.open(image_file)
hdu_list.info()
image_data = hdu_list[0].data
header = hdu_list[0].header
print(header)