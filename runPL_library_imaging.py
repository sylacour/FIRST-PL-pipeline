#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
#%%
"""
Created on Sun May 24 22:56:25 2015

@author: slacour
"""
import os
import numpy as np
from scipy import linalg
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages

def create_movie_cross(datacube):

    Nwave,Noutput,Ncube,Npos=datacube.shape
    all_flux=datacube.transpose((2,0,1,3)).reshape((Ncube,Nwave*Noutput,Npos))

    all_Minv=[]
    all_Ut=[]
    Nsingular=76
    # Nsingular=19

    for flux in tqdm(all_flux):
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
        fit_flux+=[np.matmul(linalg.pinv(Minv),images[-1])]
    mp=np.array(mp)
    images=np.array(images)
    fit_flux=np.array(fit_flux)

    residuals=fit_flux-all_flux
    residuals_std=np.std(residuals,axis=2)

    Npts=np.sqrt(images.shape[-1]).astype(int)
    Ncmap=images.shape[0]
    images=images.reshape((Ncmap*Ncmap,Npts,Npts,-1))
    images/=images.max(axis=(1,2,3))[:,None,None,None]

    print("Making movie ... ")

    def make_image(images,i):
            return images[:,:,i]

    Image=make_image(images[0],0)

    fig, axs = plt.subplots(Ncmap, Ncmap, num=15, figsize=(9.25, 9.25), clear=True)
    plt.subplots_adjust(wspace=0.025, hspace=0.025, top=0.99, bottom=0.01, left=0.01, right=0.99)

    ims=[ax.imshow(Image,vmax=0.2,vmin=-0.1) for ax in axs.ravel()]
    for ax in axs.ravel():
            ax.set_axis_off()


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


def resize_and_shift(image_2d, dither_x, dither_y, sumPos, default = np.zeros(1)):

    cmap_size2 = image_2d.shape[-1]
    Npos=len(dither_x)
    Ncube = np.prod(image_2d.shape[:-2]) // Npos
    image_2d = image_2d.reshape((Ncube, Npos, cmap_size2, cmap_size2))

    delta_x = dither_x.max()-dither_x.min()
    delta_y = dither_y.max()-dither_y.min()
    cmap_size3 = cmap_size2 + max(delta_x, delta_y)
    if sumPos:
        image_2d_bigger = np.zeros((Ncube, cmap_size3, cmap_size3))
    else:
        image_2d_bigger = np.ones((Ncube, Npos, cmap_size3, cmap_size3))

    if default.ndim < image_2d.ndim:
        default = np.expand_dims(default, axis=tuple(range(default.ndim, image_2d_bigger.ndim)))

    image_2d_bigger*=default

    for i, x, y in tqdm(zip(range(Npos), dither_x, dither_y)):
        x2 = -dither_x.min()-x
        y2 = -dither_y.min()-y
        if sumPos:
            image_2d_bigger[:, x2:x2+cmap_size2, y2:y2+cmap_size2] += image_2d[:, i]
        else:
            image_2d_bigger[:, i, x2:x2+cmap_size2, y2:y2+cmap_size2] = image_2d[:, i]
    return image_2d_bigger

def reconstruct_images(projected_data,projected_data_2_image,masque,dither_x,dither_y, sumPos = True):


    Npos=len(dither_x)
    Ncube = np.prod(projected_data.shape[1:]) // Npos

    image = projected_data_2_image @ projected_data.reshape((len(projected_data),-1))
    image_2d = np.zeros((Ncube*Npos,*masque.shape))
    image_2d[:,masque] = image.T

    image_2d_bigger= resize_and_shift(image_2d, dither_x, dither_y, sumPos)

    return image_2d_bigger


def generate_plots(singular_values, chi2_delta, flux_goodData, chi2_goodData, chi2_threshold, cross_correlated_projected_data, shifted_projected_data, fluxtiptilt_2_projdata_matrix, output_dir):
    # Singular values plot

    Nsingular = fluxtiptilt_2_projdata_matrix.shape[1]
    Ncube = cross_correlated_projected_data.shape[1]
    cmap_size = cross_correlated_projected_data.shape[-1]

    energy_estimation = (singular_values)**2 / np.sum(singular_values**2)
    reverse_cumulative_energy = np.cumsum(energy_estimation[::-1])[::-1]

    plt.figure("Singular values", clear=True)
    plt.plot(1+np.arange(len(energy_estimation)), energy_estimation**.5, marker='o', label='All Singular Values')
    plt.plot(1+np.arange(Nsingular), energy_estimation[:Nsingular]**.5, marker='o', label='Selected Singular Values')
    plt.plot(1 + np.arange(len(reverse_cumulative_energy)), reverse_cumulative_energy**.5, marker='D', label='Reverse Cumulative Energy', alpha=0.5)
    plt.plot(1+np.arange(Nsingular), reverse_cumulative_energy[:Nsingular]**.5, marker='D', alpha=0.5)

    plt.legend()
    plt.xlabel('Singular Vector Index')
    plt.ylabel('Energy Estimation')
    plt.title('Amplitude of Singular Values')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)

    # Chi2 maps plots
    fig, axs = plt.subplots(5, 1, num="reduced chi23", clear=True, figsize=(15, 20))

    chi2_delta = chi2_delta.reshape((Ncube, -1))

    axs[0].imshow(chi2_delta, aspect="auto", interpolation='none')
    axs[0].set_ylabel('Chi2')
    axs[0].set_title('Chi2 Delta')

    axs[1].imshow(flux_goodData.reshape((Ncube, -1)), aspect="auto", interpolation='none')
    axs[1].set_ylabel('N cube')
    axs[1].set_title('Masque on flux')

    axs[2].imshow(chi2_goodData.reshape((Ncube, -1)), aspect="auto", interpolation='none')
    axs[2].set_ylabel('N cube')
    axs[2].set_title('Masque on chi2')

    axs[3].plot(chi2_delta.T)
    axs[3].plot(np.ones(cmap_size * cmap_size) * chi2_threshold, 'r')
    axs[3].set_ylabel('N cube')
    axs[3].set_title('Chi2 Delta Plot')
    axs[3].set_xlim((0, cmap_size * cmap_size))

    axs[4].set_axis_off()

    axs_last = [fig.add_subplot(5, 3, 13), fig.add_subplot(5, 3, 14), fig.add_subplot(5, 3, 15)]

    max_chi2 = np.nanmax(chi2_delta.ravel())
    axs_last[0].hist(chi2_delta.ravel()[flux_goodData], bins=30, range=(0, max_chi2))
    axs_last[0].hist(chi2_delta.ravel()[chi2_goodData], bins=30, range=(0, max_chi2))
    axs_last[0].set_title('Chi2 Delta Histogram')

    axs_last[1].imshow(np.nansum(chi2_delta.reshape((Ncube, cmap_size, cmap_size)), axis=0), interpolation='none', vmin=0, vmax=max_chi2)
    axs_last[1].set_title('Chi2 Delta Sum')

    axs_last[2].imshow(chi2_goodData.reshape((Ncube, cmap_size, cmap_size)).sum(axis=0), interpolation='none')
    axs_last[2].set_title('Chi2 Good Data Sum')

    plt.tight_layout()

    # Cross-correlation matrix plot
    fig, axs = plt.subplots(Ncube, Ncube, num='cross_correlation_singular_vector', clear=True, figsize=(5, 5), squeeze=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle('Cross-correlation of singular vectors')

    for j in np.arange(Ncube):
        for k in range(Ncube):
            if j == 0:
                axs[j, k].set_title(f"{k}")
            if k == 0:
                axs[j, k].set_ylabel(f"{j}")
            axs[j, k].imshow(cross_correlated_projected_data[:, j, k].mean(axis=0), interpolation=None)
            if j != Ncube - 1:
                axs[j, k].set_xticks([])
            axs[j, k].set_yticks([])

    # Shifted singular vectors plot
    per_row = 19
    n_plots = int(np.ceil(Nsingular / per_row))
    for p in range(n_plots):
        fig, axs = plt.subplots(Ncube, per_row, num='Position map of singular vector %i to %i' % (1 + p * per_row, (1 + p) * per_row), figsize=(14 + 1, 1 + 14 * Ncube / per_row), clear=True, squeeze=False)
        plt.subplots_adjust(wspace=0, hspace=0, top=0.99, bottom=0.01, left=0.01, right=0.99)

        for i in range(p * per_row, (1 + p) * per_row):
            position_map = shifted_projected_data[i]
            for j in range(Ncube):
                vmax = np.nanmax(np.abs(position_map[j]))
                vmin = -vmax
                ax = axs[j, i % per_row]
                im = ax.imshow(position_map[j], origin='lower', cmap='viridis', vmax=vmax, vmin=vmin)
                ax.set_xticks([])
                ax.set_yticks([])
            ax.text(0.5, 0.05, f'#{i + 1}', transform=ax.transAxes, ha='center')

    # Covariance and correlation matrix plot
    F2PM = fluxtiptilt_2_projdata_matrix[:, :, 0]
    cov_matrix = np.cov(F2PM)
    cor_matrix = np.corrcoef(F2PM)

    fig, ax = plt.subplots(1, 2, num='Covariance and Correlation Matrix', figsize=(12, 6), clear=True)
    cax0 = ax[0].matshow(cov_matrix, cmap='viridis')
    fig.colorbar(cax0, ax=ax[0])
    cax1 = ax[1].matshow(cor_matrix, cmap='viridis')
    fig.colorbar(cax1, ax=ax[1])
    ax[0].set_title('Covariance Matrix of Singular Vector Models')
    ax[1].set_title('Correlation Matrix of Singular Vector Models')

    # Save all plots to a PDF
    pdf_filename = os.path.join(output_dir, "plots_summary.pdf")
    with PdfPages(pdf_filename) as pdf:
        for i in plt.get_fignums():
            fig = plt.figure(i)
            pdf.savefig(fig)

    print(f"All plots saved to {pdf_filename}")
