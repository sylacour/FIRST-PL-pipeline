import numpy as np
import os
import time
from datetime import datetime, timedelta


### plot modules
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
import matplotlib as mpl
### I/O modules
from astropy.io import fits
from scipy import ndimage
from scipy import linalg

## Shared memory function
# from pyMilk.interfacing.isio_shmlib import SHM as shm

### fitting modules
from lmfit import Model, Parameters
from lmfit.models import GaussianModel
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit

# import image_operation_functions as iof
import peakutils
from tqdm import tqdm, trange
import utils as ut

### config parameters modules
from configobj import ConfigObj
from validate import Validator

import pdb
import json



plt.ion()


__author__ = "Sebastien Vievard"
__version__ = "1.1"
__email__ = "vievard@naoj.org"



class PL_Datareconstruct(object):
    def __init__(self, config_filename, dwd):
        ##### LOAD PARAMETERS FROM CONFIG FILE #####
        configspec_file     = 'configspec_first-pl.ini'
        config              = ConfigObj(config_filename,
                                        configspec = configspec_file)
        vtor                = Validator()
        checks              = config.validate(vtor, copy=True)

        ## Coupling map info
        self.coupling_map_date      = config['coupling_map_date']
        self.coupling_map_time      = config['coupling_map_time']
        self.coupling_map_target    = config['coupling_map_target']
        self.coupling_map_path      = dwd+self.coupling_map_date+'/'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'/'
        self.coupling_map_type      = config['coupling_map_type']
        self.fratio                 = config['fratio']



#######################################################################################################################
############################################# COUPLING MAP RECONSTRUCTION #############################################
#######################################################################################################################


    def reconstruct_coupling_map(self, display = False):
        
        filename = 'info_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.txt'
        dataname = 'im_cube_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.fits'
        darkname = 'im_dark_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.fits'
        
        def spiral_function(a):####
            sp = np.zeros((a**2,2))
            switch = np.zeros(a**2)
            x = y = 0
            dx = 0
            dy = -1
            for i in range(a**2):
                sp[i,:] = np.array([x,y], dtype=float)/(a-1)*2
                if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                    switch[i] = True
                    dx, dy = -dy, dx
                else:
                    switch[i] = False
                x, y = x+dx, y+dy
            return sp,switch


        data = fits.getdata(self.coupling_map_path+dataname)
        dark = fits.getdata(self.coupling_map_path+darkname)
        for ii in range(data.shape[0]):
            data[ii] -= dark

        # Read the text file
        with open(self.coupling_map_path+filename, 'r') as file:
            data_txt = json.load(file)
        
        for key, value in data_txt.items():
            globals()[key] = value

        npt             = int(globals()['Number points'])
        window          = int(globals()['Windows size (steps)'])
        x0              = float(globals()['x init'])
        y0              = float(globals()['y init'])
        x_opt           = float(globals()['x optimal'])
        y_opt           = float(globals()['x optimal'])
        indx0           = int(npt//2.)
        indy0           = int(npt//2.)
        sp, switch      = spiral_function(npt)
        sp              = sp*(window/2.)

        window_mm       = window*0.047
        step_mm         = window_mm/npt
        wl_plot         = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
        coord_step      = np.linspace(-step_mm/2,step_mm/2,npt)

        coupling_cube   = np.zeros([data.shape[2],npt,npt])


        #### Make reconstruction if RASTER
        if self.coupling_map_type == 'raster':
            for l in tqdm(range(data.shape[2])):
                n=0
                for i in range(npt):
                    for j in range(npt):
                        coupling_cube[l,i,j] = np.sum(data[n,:,l],axis=0)
                        n+=1

        #### Make reconstruction if SPIRAL
        if self.coupling_map_type == 'spiral':       
            for l in tqdm(range(data.shape[2])):
                for i in range(0,npt**2-1):
                    xi               = np.around(x0+sp[i,0])
                    yi               = np.around(y0+sp[i,1])
                    indx             = indx0+int(sp[i,0]/(window/2.)*(npt//2.))
                    indy             = indy0+int(sp[i,1]/(window/2.)*(npt//2.))
                    coupling_cube[l,indx,indy] = np.sum(data[i,:,l],axis=0)


        if display is True:
            plt.figure(figsize=[25,5])
            for i in range(8):
                plt.subplot(1,9,i+1)
                plt.imshow(coupling_cube[wl_plot[i]], origin = 'lower', 
                            extent=[coord_step[0],coord_step[-1],coord_step[0],coord_step[-1]])
                if i == 0:
                    plt.ylabel('$\mu$m')
                plt.xlabel('$\mu$m')
            plt.tight_layout()

        return coupling_cube


    def reconstruct_coupling_map_per_output(self, dr, display = False):
        
        dr.flatfield_calibration()
        
        filename = 'info_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.txt'
        dataname = 'im_cube_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.fits'
        darkname = 'im_dark_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.fits'
        
        def spiral_function(a):####
            sp = np.zeros((a**2,2))
            switch = np.zeros(a**2)
            x = y = 0
            dx = 0
            dy = -1
            for i in range(a**2):
                sp[i,:] = np.array([x,y], dtype=float)/(a-1)*2
                if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                    switch[i] = True
                    dx, dy = -dy, dx
                else:
                    switch[i] = False
                x, y = x+dx, y+dy
            return sp,switch


        data = dr.read_data(self.coupling_map_path+dataname, extract_spectra=True)
        dark = dr.read_data(self.coupling_map_path+darkname, extract_spectra=True)
        # data = fits.getdata(self.coupling_map_path+dataname)
        # dark = fits.getdata(self.coupling_map_path+darkname)
        for ii in range(data.shape[0]):
            data[ii] -= dark

        # Read the text file
        with open(self.coupling_map_path+filename, 'r') as file:
            data_txt = json.load(file)
        
        for key, value in data_txt.items():
            globals()[key] = value

        npt             = int(globals()['Number points'])
        window          = int(globals()['Windows size (steps)'])
        x0              = float(globals()['x init'])
        y0              = float(globals()['y init'])
        x_opt           = float(globals()['x optimal'])
        y_opt           = float(globals()['x optimal'])
        indx0           = int(npt//2.)
        indy0           = int(npt//2.)
        sp, switch      = spiral_function(npt)
        sp              = sp*(window/2.)
        pl_outputs_nb   = 38

        window_mm       = window*0.047
        step_mm         = window_mm/npt
        wl_plot         = np.array([200, 400, 600, 800, 1000])
        coord_step      = np.linspace(-step_mm/2,step_mm/2,npt)

        optim_maps_outputs   = np.zeros([data.shape[2],npt,npt,pl_outputs_nb//2])

        if self.coupling_map_type == 'raster':
            for l in tqdm(range(data.shape[2])):
                n=0
                for indx in range(npt):
                    for indy in range(npt):
                        for k in range(pl_outputs_nb//2):
                            optim_maps_outputs[l,indx,indy,k] = data[n,k,l]
                        n+=1       

        if self.coupling_map_type == 'spiral':      
            for l in tqdm(range(data.shape[2])):
                for i in range(0,npt**2-1):
                    for k in range(pl_outputs_nb//2):
                        xi               = np.around(x0+sp[i,0])
                        yi               = np.around(y0+sp[i,1])
                        indx             = indx0+int(sp[i,0]/(window/2.)*(npt//2.))
                        indy             = indy0+int(sp[i,1]/(window/2.)*(npt//2.))
                        optim_maps_outputs[l,indx,indy,k] = data[i,k,l]

        if display is True:
            font0 = FontProperties()
            font = font0.copy()
            font.set_size('large')
            font.set_weight('bold')

            count=1
            # plt.figure()
            fig = plt.figure(figsize=(12.5, 23))
            # plt.subplots(7,3,figsize=[15,15])
            # plt.subplots_adjust(hspace=0)
            for j in range(0,19):
                plt.subplot(7,3,count)
                # circle = Circle(center, radius, facecolor='None', edgecolor='w', lw=1, linestyle=':',alpha=0.5)
                # circle2 = Circle(center, (25 + 15*2.44*wl_pix_map[wl_plot[i]]*10**(-3))/2 , facecolor='None', edgecolor='w', linestyle=':', lw=3) # rayon (MFD + PSF) = (25+2.44*lambda*Fratio)/2 
                plt.imshow(optim_maps_outputs[wl_plot[0],:,:,j], origin = 'lower', extent=[coord_step[0],coord_step[-1],coord_step[0],coord_step[-1]])
                # plt.gca().add_patch(circle)
                # plt.gca().add_patch(circle2)
                # if j == [0,5,10,15]:
                plt.ylabel('$\mu$m')
                # plt.title('$\lambda$ = '+str(np.round(wl_pix_map[wl_plot[i]], decimals = 2))+' nm')
                # if j == [15,16,17,18,19]:
                plt.xlabel('$\mu$m')
                plt.clim(np.min(optim_maps_outputs[wl_plot[0]]), np.max(optim_maps_outputs[wl_plot[0]]))
                count = count + 1
                # plt.text(-6,11,f'Output {j+1}',fontproperties=font,ha='center',color='white')
                cbar=plt.colorbar()
                cbar.formatter.set_powerlimits((4, 4))

            plt.subplot(7,3,count)
            # circle = Circle(center, radius, facecolor='None', edgecolor='w', lw=1, linestyle=':',alpha=0.5)
            plt.imshow(np.sum(optim_maps_outputs[wl_plot[0]],axis=2), origin = 'lower', extent=[coord_step[0],coord_step[-1],coord_step[0],coord_step[-1]])
            # plt.gca().add_patch(circle)
            # plt.ylabel(f'Output {j}')
            # plt.text(-9,11,'Total',fontproperties=font,ha='center',color='white')
            plt.xlabel('$\mu$m')
            plt.ylabel('$\mu$m')
            cbar=plt.colorbar()
            cbar.formatter.set_powerlimits((0, 4))
            plt.tight_layout()

        return optim_maps_outputs


#######################################################################################################################
################################################## IMAGE RECONSTRUCTION ###############################################
#######################################################################################################################

    def make_F2IM(self,dr,display=False):

        dr.flatfield_calibration()
        
        filename = 'info_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.txt'
        dataname = 'im_cube_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.fits'
        darkname = 'im_dark_'+self.coupling_map_date+'_'+self.coupling_map_time+'_'+self.coupling_map_target+'.fits'
        


        data = dr.read_data(self.coupling_map_path+dataname, extract_spectra=True)
        dark = dr.read_data(self.coupling_map_path+darkname, extract_spectra=True)
        # data = fits.getdata(self.coupling_map_path+dataname)
        # dark = fits.getdata(self.coupling_map_path+darkname)
        for ii in range(data.shape[0]):
            data[ii] -= dark

        # Read the text file
        with open(self.coupling_map_path+filename, 'r') as file:
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

        spectra = dr.extract_trace_spectra(data)

        polar_1 = np.array([0,1,2,4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,32,34]) 
        polar_2 = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,36,37]) 

        flux = []
        for i in range(19):
            flux.append(spectra[:,polar_1[i],:]+spectra[:,polar_2[i],:])

        flux = np.array(flux)#.reshape([flux.shape[1], 19, flux.shape[2]])
        flux = np.reshape(flux,[flux.shape[1], 19, flux.shape[2]])


        ## NEED NORMALIZATION

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

        self.F2IM       = np.array(M_inv)
        U               = np.array(U_T)
        s               = np.array(s_T)
        Vh              = np.array(Vh_T).reshape((-1,19,npt,npt))

        return self.F2IM



    def reconstruct_image(self,flux,dr,display = False):

        # Init reconstruction images
        recons_img = []

        # Loop projecting fluxes on F2IM
        for i in range(flux.shape[1]):
            Wavelength_channel = i
            F2IM_channel=self.F2IM[Wavelength_channel]
            recons_img.append(np.dot(F2IM_channel,flux[:,Wavelength_channel]))

        recons_img = np.array(recons_img)
        recons_img = recons_img.reshape((flux.shape[1],npt,npt))

        if display is True :
            recons_img_mean = np.mean(recons_img,axis=0)
            plt.figure(1)
            plt.clf()
            plt.imshow(recons_img_mean, origin='lower', extent=[coord_mas[0],coord_mas[-1],coord_mas[0],coord_mas[-1]])
            plt.title('Image reconstruction')

        return recons_img


#######################################################################################################################
############################################### SPECTRUM RECONSTRUCTION ###############################################
#######################################################################################################################

    def load_spectra(self, dr):
            
        
        # Times of the observation
        start_observation = datetime.strptime(dr.start_time, "%H:%M")
        end_observation = datetime.strptime(dr.end_time, "%H:%M")
        interval_minutes = 1
        num_intervals = int(((end_observation - start_observation).seconds) / 60 / interval_minutes)
        times = [(start_observation + timedelta(minutes=interval_minutes*i)).strftime("%H:%M") 
            for i in range(num_intervals + 1)]


        dir_data=dr.save_dir+'Datared/'
        for files in os.walk(dir_data): 
                for filename in files:
                        list_of_files = filename

        list_of_files.sort()
        list_of_red_files=[]
        list_of_cal_files=[]

        for file_time in times:
                for n in range(np.size(list_of_files)):
                        if (list_of_files[n])[:14] == dr.data_date+'_'+file_time and list_of_files[n][np.size(list_of_files[n])-8:np.size(list_of_files[0])-5] == 'red':
                                list_of_red_files+=[list_of_files[n]]
                        if (list_of_files[n])[:14] == dr.data_date+'_'+file_time and list_of_files[n][np.size(list_of_files[n])-8:np.size(list_of_files[0])-5] == 'cal':
                                list_of_cal_files+=[list_of_files[n]]

        print(list_of_red_files)
        print(list_of_cal_files)

        self.all_traces = []
        all_spectra = []
        all_wavecal = []

        for i in tqdm(range(np.size(list_of_red_files))):
            data_red = np.load(dr.save_dir+'Datared/'+list_of_red_files[i])
            data_cal = np.load(dr.save_dir+'Datared/'+list_of_cal_files[i])
            
            spectra = data_red['data']          # Get spectra

            ### SEB COMMENTED THIS, WAS GIVING A SEG FAULT
            # for i in np.arange(19):
            #     plt.subplot(19,2,2*i+1)
            #     if i == 0:
            #             plt.title('Polar 1')
            #     if i == 18:
            #             plt.xlabel('Wavelength (nm)')
            #     plt.plot(data_cal['wave'] ,np.mean(spectra,axis=0)[int(dr.polar1[i]),:].T,'r')
            #     plt.ylabel(str(i+1))
            #     plt.subplot(19,2,2*i+2)
            #     if i == 0:
            #             plt.title('Polar 2')
            #     if i == 18:
            #             plt.xlabel('Wavelength (nm)')
            #     plt.plot(data_cal['wave'] ,np.mean(spectra,axis=0)[int(dr.polar2[i]),:].T,'g')
            # pdb.set_trace()

            self.all_traces.extend(spectra)
            spectra = np.sum(spectra,axis=1)    # Co-add 38 spectra


            
            wavecal = data_cal['wave']          # Get wavecal

            all_spectra.extend(spectra)
            all_wavecal.extend(wavecal)

        all_spectra = np.array(all_spectra)
        self.all_traces  = np.array(self.all_traces)
        all_wavecal = wavecal
        # all_wavecal = np.array(all_wavecal)

        self.all_spectra = all_spectra

        return all_spectra, all_wavecal

    
    def spectrum_reconstruction(self, dr, wavelength_filtering = 700, threshold = 0.5, display = True):
            
        spectra, wavecal = self.load_spectra(dr)

        # wavecal = wavecal[0:1500]

        # filtering data
        select_frames = np.where(spectra[:,wavelength_filtering] > threshold*np.max(spectra[:,wavelength_filtering]))
        print('Selected '+str(np.size(select_frames))+'/'+str(spectra.shape[0]))

        selected_spectra = spectra[select_frames]

        std_spectra = np.std(selected_spectra,axis=0)
        
        reconstruct_spectrum = np.mean(selected_spectra,axis=0)
        if display is True:
            plt.rcParams.update({'font.size': 20})
            plt.figure(22,figsize=[20,5])
            plt.clf()
            plt.plot(wavecal, reconstruct_spectrum,'.r',alpha=0.5)
            # plt.plot(wavecal[0:1500], reconstruct_spectrum,'k',alpha=0.3)
            plt.title(dr.object_to_reduce+' spectrum reconstruction')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Flux (ADU)')

        for i in np.arange(19):
                plt.subplot(19,2,2*i+1)
                if i == 0:
                        plt.title('Polar 1')
                if i == 18:
                        plt.xlabel('Wavelength (nm)')
                plt.plot(selected_spectra[:,int(dr.polar1[i])].T,'r')
                plt.ylabel(str(i+1))
                plt.subplot(19,2,2*i+2)
                if i == 0:
                        plt.title('Polar 2')
                if i == 18:
                        plt.xlabel('Wavelength (nm)')
                plt.plot(selected_spectra[:,int(dr.polar2[i])].T,'g')

        return reconstruct_spectrum, wavecal

            