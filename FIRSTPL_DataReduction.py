import numpy as np
import os
import time
from datetime import datetime, timedelta



### plot modules
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
### I/O modules
from astropy.io import fits
from scipy import ndimage

## Shared memory function
# from pyMilk.interfacing.isio_shmlib import SHM as shm

### fitting modules
from lmfit import Model, Parameters
from lmfit.models import GaussianModel
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit

# import image_operation_functions as iof
import peakutils
import utils as ut

### config parameters modules
from configobj import ConfigObj
from tqdm import tqdm
from validate import Validator

import ipdb


plt.ion()


__author__ = "Sebastien Vievard"
__version__ = "1.1"
__email__ = "vievard@naoj.org"


class PL_Datareduce(object):
        def __init__(self, config_filename, dwd, savedir):
                ##### LOAD PARAMETERS FROM CONFIG FILE #####
                configspec_file     = 'configspec_first-pl.ini'
                config              = ConfigObj(config_filename,
                                                configspec = configspec_file)
                vtor                = Validator()
                checks              = config.validate(vtor, copy=True)

                config['dwd'] = dwd
                config['save_dir'] = savedir

                self.config       =     config

        def load_configuration(self):

                config = self.config
                # Main paths
                self.dwd                            = config['dwd']
                self.save_dir                       = config['save_dir']
                self.data_date                      = config['data_date']
                self.data_dir                       = self.dwd + self.data_date + '/'

                # FIRST PL info
                self.polar1                         = config['polarization_1']
                self.polar2                         = config['polarization_2']

                # Image pixel info
                self.frameheight                    = config['frameheight']
                self.framewidth                     = config['framewidth']
                self.min_pixel                      = int(config['min_pixel'])
                self.max_pixel                      = int(config['max_pixel'])
                self.extraction_width               = int(config['extraction_width'])

                # Wavelength calibration
                self.wavecal_source                 = config['wavecal_source']
                self.wavecal_date                   = config['wavecal_date']
                self.wavecal_prefix                 = config['wavecal_prefix']
                self.wavecal_dark_prefix            = config['wavecal_dark_prefix']
                self.wavelength_list                = config['wavelength_list']
                self.wavelength_threshold_up        = config['wavelength_threshold_up']
                self.wavelength_threshold_down      = config['wavelength_threshold_down']
                self.wavelength_peak_threshold      = config['wavelength_peak_threshold']

                # Flat field calibration
                self.flat_field_date                = config['flat_field_date']
                self.flat_field_prefix              = config['flat_field_prefix']
                self.flat_field_dark_prefix         = config['flat_field_dark_prefix']
                self.flat_field_object              = config['flat_field_object']
                
                # Science Target
                self.object_to_reduce               = config['object_to_reduce']
                self.start_time                     = config['start_time']
                self.end_time                       = config['end_time']
                self.object_dark_file               = config['object_dark_file']
                self.size_cube                      = config['size_cube']

                # Date / Save figures
                self.date                           = datetime.today().strftime('%Y%m%d')     
                self.save_dir                       = os.path.dirname(self.save_dir) + '/' + self.data_date+ '/'
                self.save_name                      = self.data_date

                
#         _________     _____   .____     .___ __________ __________    _____ ___________.___ ________    _______           #
#         \_   ___ \   /  _  \  |    |    |   |\______   \\______   \  /  _  \\__    ___/|   |\_____  \   \      \          #
#         /    \  \/  /  /_\  \ |    |    |   | |    |  _/ |       _/ /  /_\  \ |    |   |   | /   |   \  /   |   \         #
#         \     \____/    |    \|    |___ |   | |    |   \ |    |   \/    |    \|    |   |   |/    |    \/    |    \        #
#          \______  /\____|__  /|_______ \|___| |______  / |____|_  /\____|__  /|____|   |___|\_______  /\____|__  /        #
#                 \/         \/         \/             \/         \/         \/                       \/         \/         #

        def make_directories(self):
                if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                if not os.path.exists(self.save_dir+'Figures/'):
                        os.makedirs(self.save_dir+'Figures/')
                if not os.path.exists(self.save_dir+'FlatFields/'):
                        os.makedirs(self.save_dir+'FlatFields/')
                if not os.path.exists(self.save_dir+'Datared/'):
                        os.makedirs(self.save_dir+'Datared/')


#######################################################################################################################
############################################# FLAT-FIELD CALIBRATION ##################################################
#######################################################################################################################

        def flatfield_calibration(self, display = False, save_plots = False):
                self.load_configuration()
                #-----------------------------------------------------------------------------
                # Load images
                # -----------------------------------------------------------------------------
                print(self.dwd+self.flat_field_date+'/orcam/')
                #filename_dark=self.dwd+self.flat_field_date+'/orcam/'+self.flat_field_dark_prefix+'.fits'
                filename_dark=self.dwd+"2024-11-21/2024-11-21_13-48-32_science/im_dark_2024-11-21_13-48-32_science.fits"
                dark = self.read_data(filename_dark, mean=True)

                list_of_files=[]
                dir_flats = self.dwd+self.flat_field_date+"/"#+'/orcam/'
                for files in os.walk(dir_flats): 
                        print(files)
                        for filename in files[-1]:
                                if filename[:11] == self.flat_field_prefix and filename[-5:] == '.fits':
                                        list_of_files += [filename]

                list_of_files.sort()

                data_cube=[]
                for f in list_of_files:
                        data = self.read_data(dir_flats+f)
                        data_cube+=[data]

                data_cube=np.concatenate(data_cube)

                image   = np.mean(data_cube, axis=0) - dark
                image_var       = np.std(data_cube, axis=0)**2

                linear_coefs = np.polyfit(image.ravel(),image_var.ravel(),1)
                
                # images_finales        = Stacked_image/np.max(Stacked_image)
                # pdb.set_trace()
                self.traces_loc         = np.ones([image.shape[1],38],dtype=int)
                for i in range(15):
                        subwindow = image[:,i*100:i*100+100]
                        sum_subwindow = np.sum(subwindow,axis=1)
                        # plt.figure();plt.plot(sum_subwindow)
                        detectedWavePeaks = peakutils.peak.indexes(sum_subwindow,thres=0.06, min_dist=1)
                        arr = np.expand_dims(detectedWavePeaks,axis=0)
                        # print(detectedWavePeaks.shape)
                        if detectedWavePeaks.shape[0] == 38 : 
                                self.traces_loc[i*100:i*100+100,:]*= arr
                        else:
                                print('Error loc peaks, i='+str(i))
                                
                if display == True:
                        plt.figure("Extract windows of flat",clear=True)
                        plt.imshow(image,aspect="auto")
                        plt.plot(self.traces_loc,'r',linewidth=0.83,zorder=10,alpha=0.5)

                        plt.figure("Calculate detector gain",clear=True)
                        plt.plot(image,image_var,'k.',alpha=0.2)
                        max_value=max(image.ravel())
                        plt.plot([0,max_value],[linear_coefs[1],linear_coefs[0]*max_value],'r',label=r"$\gamma =$ %.3f ADU/e-"%linear_coefs[0])
                        plt.legend()
                        plt.xlabel("Pixel flux (ADU)")
                        plt.ylabel(r"Pixel variance (ADU$^2$)")


                self.flat_field_spectra = self.extract_trace_spectra(image, display = True)


        def flatfield_calibration_fit(self, display = False, save_plots = False):
                #-----------------------------------------------------------------------------
                # Load images
                # -----------------------------------------------------------------------------
                print(self.dwd+self.flat_field_date+'/firstpl/')
                filename_dark=self.dwd+self.flat_field_date+'/firstpl/'+self.flat_field_dark_prefix+'.fits'
                dark = self.read_data(filename_dark, mean=True)

                list_of_files=[]
                dir_flats = self.dwd+self.flat_field_date+'/firstpl/'
                for files in os.walk(dir_flats): 
                        # print(files)
                        for filename in files[-1]:
                                if filename[:13] == self.flat_field_prefix and filename[-5:] == '.fits':
                                        list_of_files += [filename]

                list_of_files.sort()

                data_cube=[]
                for f in list_of_files:
                        data = self.read_data(dir_flats+f)
                        data_cube+=[data]

                data_cube       = np.concatenate(data_cube)

                image           = np.mean(data_cube, axis=0) - dark
                image_var       = np.std(data_cube, axis=0)**2

                linear_coefs = np.polyfit(image.ravel(),image_var.ravel(),1)
                
                sampling        = np.linspace(10,int(image.shape[1]),300,dtype=int)
                peaks           = np.zeros([38, sampling.shape[0]])
                
                threshold_array=[0.01,0.015,0.02,0.03,0.05,0.08]
                detectedWavePeaks=[]
                for i in tqdm(range(sampling.shape[0])):
                        sum = image[:,sampling[i]-5:sampling[i]+5].sum(axis=1)
                        for t in threshold_array:
                                detectedWavePeaks_tmp = peakutils.peak.indexes(sum,thres=t, min_dist=6)
                                if len(detectedWavePeaks_tmp) == 38:
                                        detectedWavePeaks=detectedWavePeaks_tmp
                        if display is True:
                                plt.figure(1);plt.clf();plt.plot(sum);plt.plot(detectedWavePeaks,sum[detectedWavePeaks],'+r');plt.draw();plt.pause(0.0000001)
                        
                        peaks[:,i]              = detectedWavePeaks
                
                self.traces_loc         = np.ones([image.shape[1],38],dtype=int)

                for i in range(38):
                        x = sampling
                        y = peaks[i]
                        coefficients = np.polyfit(x, y, 1)
                        p=np.poly1d(coefficients)
                        self.traces_loc[:,i] = p(np.arange(image.shape[1]))

                                
                if display == True:
                        plt.figure("Extract fitted traces",clear=True)
                        plt.imshow(image,aspect="auto")
                        for i in range(38): 
                                plt.plot(self.traces_loc[:,i],',r')
                        # plt.plot(self.traces_loc,'r',linewidth=0.83,zorder=10,alpha=0.5)

                        plt.figure("Calculate detector gain",clear=True)
                        plt.plot(image,image_var,'k.',alpha=0.2)
                        max_value=max(image.ravel())
                        plt.plot([0,max_value],[linear_coefs[1],linear_coefs[0]*max_value],'r',label=r"$\gamma =$ %.3f ADU/e-"%linear_coefs[0])
                        plt.legend()
                        plt.xlabel("Pixel flux (ADU)")
                        plt.ylabel(r"Pixel variance (ADU$^2$)")


                self.flat_field_spectra = self.extract_trace_spectra(image, display = False)




#######################################################################################################################
############################################# WAVELENGTH CALIBRATION ##################################################
#######################################################################################################################


        def wavelength_calibration(self, display = False, save_plots = False, redo = False):

                if (os.path.isfile(self.save_dir+self.wavecal_prefix+'Wavecal_tab.npy')&(redo == False)) is True :#and os.path.isfile(self.save_dir+self.wavecal_prefix+'Selpix_tab.npy') is True :
                        print('*** WAVECAL : LOADING PREVIOUS WAVELENGTH CALIBRATION')
                        self.pix_to_wavelength_map  = np.load(self.save_dir+self.wavecal_prefix+'Wavecal_tab.npy')
                        # self.selpix                 = np.load(self.save_dir+self.wavecal_prefix+'Selpix_tab.npy')
                        self.n_pixofinterest        = len(self.pix_to_wavelength_map)
                        # self.curvature_correction_get_curve(load=True)

                # if os.path.isfile('/home/first/sebviev/Data/PL/20230730_PL_Neon_wavelengthcal.npy') is True :
                #         pix_to_wavelength_map = np.load('/home/first/sebviev/Data/PL/20230730_PL_Neon_wavelengthcal.npy')

                        # return pix_to_wavelength_map
                else:
                        # Get wavecal files

                        filename_neon = self.dwd+self.wavecal_date+'/firstpl/'+self.wavecal_prefix+'.fits'
                        data_neon = self.read_data(filename_neon, mean = True, extract_spectra = True)

                        filename_neon_dark = self.dwd+self.wavecal_date+'/firstpl/'+self.wavecal_dark_prefix+'.fits'
                        dark_neon = self.read_data(filename_neon_dark, mean = True, extract_spectra = True)

                        # Get Dark-subtracted data
                        img = data_neon - dark_neon


                        if display is True :
                                plt.figure()
                                plt.imshow(np.log10(img+abs(2*np.min(img))), cmap='inferno',aspect='auto')
                                plt.xlabel("Wavelength")
                                plt.ylabel("Photonic Lantern Output")

                        # Extract spectra
                        self.output_spectra_wl = img.mean(axis=0)
                        self.pix_to_wavelength_map = np.zeros_like(self.output_spectra_wl)

                                
                        detectedWavePeaks = peakutils.peak.indexes(self.output_spectra_wl,thres=self.wavelength_peak_threshold, min_dist=12)

                        plt.figure();plt.plot(self.output_spectra_wl)
                        plt.plot(detectedWavePeaks, self.output_spectra_wl[detectedWavePeaks], 'or')
                        # print(detectedWavePeaks.shape)
                        print(detectedWavePeaks)
                        self.detectedWavePeaks = detectedWavePeaks
                                # stop
                                # if display is True :
                                #         plt.figure()
                                #         plt.plot(spectra)
                                #         plt.plot(detectedWavePeaks, spectra[detectedWavePeaks], 'or')
                                #         plt.xlabel('Wavelength axis')

                        WavePoly=np.polyfit(detectedWavePeaks,self.wavelength_list,2)
                        Wavefit=np.poly1d(WavePoly)
                        self.pix_to_wavelength_map=Wavefit(np.arange(0,img.shape[1]))
                        

                        if display is True :
                                fig1, ax = plt.subplots(num=11,figsize=(6,6))
                                ax.plot(np.arange(0,img.shape[1]), self.pix_to_wavelength_map,
                                        label='Polynomial fit (deg={})'.format(2))
                                ax.plot(detectedWavePeaks, self.wavelength_list, 'x', ms=6,
                                        label='Detected peaks')
                                ax.set_xlabel("Pixel number")
                                ax.set_ylabel("Wavelength axis")
                                ax.legend()


                        if display is True:
                                fig1, ax = plt.subplots(2,1,num=12,figsize=(6,6))
                                # fig1.subplot(1,2,1)
                                ax[0].plot(self.output_spectra_wl)
                                ax[0].plot(detectedWavePeaks, self.output_spectra_wl[detectedWavePeaks], 'or')
                                ax[0].set_yscale('log') 
                                ax[1].plot(np.arange(0,img.shape[1]), self.pix_to_wavelength_map,
                                        label='Polynomial fit (deg={})'.format(2))
                                ax[1].plot(detectedWavePeaks, self.wavelength_list, 'x', ms=6,
                                        label='Detected peaks')
                                ax[1].set_xlabel("Pixel number")
                                ax[1].set_ylabel("Wavelength axis")
                                ax[1].legend()                               


                        np.save(self.save_dir+self.wavecal_prefix+'Wavecal_tab.npy', self.pix_to_wavelength_map)


        def resolution_power(self,display = False, save_plots = False): 
                        # Get wavecal files
                        file_neon = fits.getdata(self.dwd+self.wavecal_date+'/'+self.wavecal_prefix+'.fits')
                        file_neon_dark = fits.getdata(self.dwd+self.wavecal_date+'/'+self.wavecal_dark_prefix+'.fits')

                        # Get Dark-subtracted data
                        img = np.mean(file_neon, axis=0) - np.mean(file_neon_dark, axis=0)

                        # Rotate images
                        img = ut.cent_rot(img,0.3,np.array((int(img.shape[0]/2.),int(img.shape[1]/2.))))

                        if display is True :
                                plt.figure()
                                plt.imshow(np.log10(img+abs(2*np.min(img))), cmap='inferno')
                                plt.xlabel("Wavelength")
                                plt.ylabel("Photonic Lantern Output")


                        spectra = self.extract_trace_spectra(img, display = True, save_plots= True)

                        def gauss(x, amp,cen,wid):
                                return amp*np.exp(-(x-cen)**2/(2.*wid**2))


                        output_res = []
                        sum_spectrum_total      = np.sum(img, axis=0) ## sum on wavelength dimensions --> should give 38 peaks
                        detectedWavePeaks       = peakutils.peak.indexes(sum_spectrum_total,thres=0.1, min_dist=1) # Detect peaks
                        plt.figure()
                        plt.plot(sum_spectrum_total)
                        plt.plot(detectedWavePeaks, sum_spectrum_total[detectedWavePeaks], 'or')
                        plt.xlabel('Wavelength axis')
                        plt.figure()
                        for i in np.arange(38):
                                Spectrum = spectra[i]
                                inst_res=[]
                                for ii in range(detectedWavePeaks.size):
                                        temp=np.zeros(Spectrum.size)
                                        temp[detectedWavePeaks[ii]-10:detectedWavePeaks[ii]+10] = Spectrum[detectedWavePeaks[ii]-10:detectedWavePeaks[ii]+10]

                                        popt, pcov = curve_fit(gauss, self.pix_to_wavelength_map, temp, p0=(Spectrum[detectedWavePeaks[ii]],self.pix_to_wavelength_map[detectedWavePeaks[ii]], 2))

                                        temp_fit = gauss(np.arange(600,800,0.1), popt[0], popt[1], popt[2])
                                        # if i ==0:
                                        #         plt.figure()
                                        #         plt.plot(np.arange(600,800,0.1),temp_fit)
                                        #         plt.plot(self.pix_to_wavelength_map,temp,'or')

                                        FWHM = 2*((2*np.log(2))**0.5)*abs(popt[2])
                                        inst_res.append(popt[1]/FWHM)

                                inst_res = np.array(inst_res)

                                output_res.append(inst_res)
                                # 
                                plt.plot(self.pix_to_wavelength_map[detectedWavePeaks], inst_res, 'o', markersize=2)
                                plt.plot(self.pix_to_wavelength_map[detectedWavePeaks], inst_res, '--')
                                plt.ylabel('PL spectral resolution')
                                plt.xlabel('Wavelength (nm)')
                                
                                
                        output_res = np.array(output_res)
                        output_res_mean = np.mean(output_res,axis =0)

                        plt.figure()
                        plt.plot(self.pix_to_wavelength_map[detectedWavePeaks], output_res_mean, 'o', markersize=2)
                        plt.plot(self.pix_to_wavelength_map[detectedWavePeaks], output_res_mean, '--')
                        plt.ylabel('PL spectral resolution')
                        plt.xlabel('Wavelength (nm)')

                        
#                                                                                                                                  #
#    ________      _____ ___________ _____      ___ ___     _____    _______   ________   .____     .___  _______     ________     #
#    \______ \    /  _  \\__    ___//  _  \    /   |   \   /  _  \   \      \  \______ \  |    |    |   | \      \   /  _____/     #
#     |    |  \  /  /_\  \ |    |  /  /_\  \  /    ~    \ /  /_\  \  /   |   \  |    |  \ |    |    |   | /   |   \ /   \  ___     #
#     |    `   \/    |    \|    | /    |    \ \    Y    //    |    \/    |    \ |    `   \|    |___ |   |/    |    \\    \_\  \    #
#    /_______  /\____|__  /|____| \____|__  /  \___|_  / \____|__  /\____|__  //_______  /|_______ \|___|\____|__  / \______  /    #
#            \/         \/                \/         \/          \/         \/         \/         \/             \/         \/     #
#                                                                                                                                  #



################################################################################################################################
######################################################## GET IMAGES ############################################################
################################################################################################################################

        def save_images(self, name_shared_mem = 'orcam', cube_average = False, display = False, save_plots = False):
                
                #-----------------------------------------------------------------------------
                # Load dark
                # -----------------------------------------------------------------------------
                filename_dark=self.dwd+self.data_date+'/firstpl/'+self.object_dark_file+'.fits'
                dark, dark_err = self.read_data(filename_dark, mean = True, get_std = True, extract_spectra= True)


                #-----------------------------------------------------------------------------
                # Load images
                # -----------------------------------------------------------------------------

                # Times of the observation
                start_observation = datetime.strptime(self.start_time, "%H:%M")
                end_observation = datetime.strptime(self.end_time, "%H:%M")
                interval_minutes = 1
                num_intervals = int(((end_observation - start_observation).seconds) / 60 / interval_minutes)
                times = [(start_observation + timedelta(minutes=interval_minutes*i)).strftime("%H:%M") 
                   for i in range(num_intervals + 1)]


                dir_data=self.dwd+self.data_date+'/firstpl/'
                for files in os.walk(dir_data): 
                        for filename in files:
                                list_of_files = filename

                list_of_files.sort()
                list_of_good_files=[]

                for file_time in times:
                        for n in range(np.size(list_of_files)):
                                if (list_of_files[n])[:11] == 'orcam_'+file_time and (list_of_files[n])[np.size(list_of_files[n])-5:] == 'fits':
                                        list_of_good_files+=[list_of_files[n]]


                if name_shared_mem == 'firstpl' :
                        for file_time in times:
                                for n in range(np.size(list_of_files)):
                                        if (list_of_files[n])[:13] == 'firstpl_'+file_time and (list_of_files[n])[np.size(list_of_files[n])-5:] == 'fits':
                                                list_of_good_files+=[list_of_files[n]]

                        print(list_of_good_files)

                        flat = self.flat_field_spectra
                        wave = self.pix_to_wavelength_map
                        spectra_wl =self.output_spectra_wl
                        data_avg=[]
                        for filename in list_of_good_files[:]:
                                data =  self.read_data(dir_data+filename,  extract_spectra= True)-dark 
                                data_avg += [data.sum(axis=0)]
                                if cube_average:
                                        data = data_avg[-1]
                                fits_header=fits.getheader(dir_data+filename)
                                print('writing : ',self.save_dir+'Datared/'+self.data_date+filename[7:-5]+'_red.npz')
                                np.savez_compressed(self.save_dir+'Datared/'+self.data_date+filename[7:-5]+'_red.npz', data=data)
                                np.savez_compressed(self.save_dir+'Datared/'+self.data_date+filename[7:-5]+'_cal.npz',  header=dict(fits_header), config = dict(self.config), wave=wave, dark=dark, dark_err=dark_err, flat = flat, spectra_wl = spectra_wl, traces_loc=self.traces_loc)
                
                ### BELOW, WILL DISAPEAR SOON
                if name_shared_mem == 'orcam' :
                        for file_time in times:
                                for n in range(np.size(list_of_files)):
                                        if (list_of_files[n])[:11] == 'orcam_'+file_time and (list_of_files[n])[np.size(list_of_files[n])-5:] == 'fits':
                                                list_of_good_files+=[list_of_files[n]]

                        print(list_of_good_files)

                        flat = self.flat_field_spectra
                        wave = self.pix_to_wavelength_map
                        spectra_wl =self.output_spectra_wl
                        data_avg=[]
                        for filename in list_of_good_files[:]:
                                data =  self.read_data(dir_data+filename,  extract_spectra= True)-dark 
                                data_avg += [data.sum(axis=0)]
                                if cube_average:
                                        data = data_avg[-1]
                                fits_header=fits.getheader(dir_data+filename)
                                print('writing : ',self.save_dir+'Datared/'+self.data_date+filename[5:-5]+'_red.npz')
                                np.savez_compressed(self.save_dir+'Datared/'+self.data_date+filename[5:-5]+'_red.npz', data=data)
                                np.savez_compressed(self.save_dir+'Datared/'+self.data_date+filename[5:-5]+'_cal.npz',  header=dict(fits_header), config = dict(self.config), wave=wave, dark=dark, dark_err=dark_err, flat = flat, spectra_wl = spectra_wl, traces_loc=self.traces_loc)

                data_summed = np.array(data_avg).sum(axis=0)
                print(data_summed.shape)

                #-----------------------------------------------------------------------------
                # Display images
                # -----------------------------------------------------------------------------
                if display is True:
                        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                                ['black','green','white'],
                                                                                256)

                        plt.rcParams.update({'font.size': 20})
                        # e = [self.pix_to_wavelength_map[-1], self.pix_to_wavelength_map[0],0, np.shape(data_summed)[0]/8.]
                        plt.figure(figsize = [20,5])
                        # plt.imshow(np.fliplr(np.log10(images_finales+2*np.min(images_finales))), cmap=cmap, extent=e)
                        plt.imshow(np.fliplr(data_summed), cmap=cmap,aspect="auto")
                        plt.gca().axes.yaxis.set_ticklabels([])
                        plt.title('Humu spectra acquisition')
                        plt.xlabel('Wavelength (nm)')
                        plt.ylabel('Outputs')
                        plt.colorbar()
                        if save_plots is True:
                                plt.savefig(self.save_dir+'Figures/'+self.object_to_reduce+'_'+self.data_date+'_Averaged_image.pdf', dpi=300,bbox_inches='tight', pad_inches=0.1, format='pdf')

                return data_summed



################################################################################################################################
#################################################### SPECTRA EXTRACTION ########################################################
################################################################################################################################

        def read_data(self, filename, mean = False, get_std = False, extract_spectra = False, display = False):

                print("reading : "+filename)
                data = fits.getdata(filename)
                if np.size(data.shape) == 3:
                        data = data[:,:,self.min_pixel:self.max_pixel]
                if np.size(data.shape) == 2:
                        data = data[:,self.min_pixel:self.max_pixel]
                if get_std:
                        data_std = data.std(axis=0)/np.sqrt(len(data))
                if mean:
                        data = data.mean(axis=0)
                
                if extract_spectra:
                        data = self.extract_trace_spectra(data, display = display)

                if get_std:
                        return data,data_std
                else:
                        return data


        def extract_trace_spectra(self, image, display = False):
                nb_outputs = 38
                Nside = self.extraction_width 
                image_shape = image.shape
                if len(image_shape) == 2:
                        image = np.expand_dims(image,axis=0)
                Ndit = image.shape[0]
                Nwidth = image.shape[2]
                out_spectra  = np.zeros([Ndit,nb_outputs,Nwidth]) ## Prep vector with all traces (8 pix thick)
                out_image  = np.zeros([Nwidth,nb_outputs,Nside*2+2])


                # ipdb.set_trace()
                for i in range(Nwidth):
                        for j in range(nb_outputs):
                                out_spectra[:,j,i] = image[:,int(self.traces_loc[i,j])-Nside:int(self.traces_loc[i,j])+Nside+1,i].sum(axis=1)
                                res3 = int(self.traces_loc[i,j])+Nside+1
                                res2 = int(self.traces_loc[i,j])-Nside
                                res1 = image[:,res2:res3,i] #res3 - res2 = 9 -> res1 shape should be (1,9)
                                res = res1.sum(axis=0) #shape 9
                                out_image[i,j,:Nside*2+1] = res

                ## Check the output spectra
                if display is True :
                        plt.figure("spectras on detector",clear=True)
                        plt.imshow(out_image[:,:].reshape((Nwidth,-1)).T,aspect='auto',interpolation='none')

                        plt.subplots(19,2,sharex=True,sharey=True,figsize=[10,20])
                        plt.subplots_adjust(hspace=0)
                        for i in np.arange(19):
                                plt.subplot(19,2,2*i+1)
                                if i == 0:
                                        plt.title('Polar 1')
                                if i == 18:
                                        plt.xlabel('Wavelength (nm)')
                                plt.plot(out_spectra[:,int(self.polar1[i])].T,'r')
                                plt.ylabel(str(i+1))
                                plt.subplot(19,2,2*i+2)
                                if i == 0:
                                        plt.title('Polar 2')
                                if i == 18:
                                        plt.xlabel('Wavelength (nm)')
                                plt.plot(out_spectra[:,int(self.polar2[i])].T,'g')

                if Ndit == 1:
                        return out_spectra[0]
                else:
                        return out_spectra


        def make_trace_loc(self,image, display = False):
                

                self.traces_loc         = np.ones([image.shape[1],38],dtype=int)
                for i in range(15):
                        subwindow = image[:,i*100:i*100+100]
                        sum_subwindow = np.sum(subwindow,axis=1)
                        # plt.figure();plt.plot(sum_subwindow)
                        detectedWavePeaks = peakutils.peak.indexes(sum_subwindow,thres=0.02, min_dist=1)
                        arr = np.expand_dims(detectedWavePeaks,axis=0)
                        # print(detectedWavePeaks.shape)
                        if detectedWavePeaks.shape[0] == 38 : 
                                self.traces_loc[i*100:i*100+100,:]*= arr
                        else:
                                print('Error loc peaks, i='+str(i))

                if display == True:
                        plt.figure("Extract windows of flat",clear=True)
                        plt.imshow(image,aspect="auto")
                        plt.plot(self.traces_loc,'r',linewidth=2,zorder=10)
