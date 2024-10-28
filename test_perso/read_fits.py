import numpy as np

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


def getdata(source):
    hdu_list = fits.open(source)
    image_data = hdu_list[0].data
    return image_data

neon1 = getdata("C:/Users/Jehanne/Desktop/Doctorat/FIRST-PL/Neon1/PL_Neon.fits")
darkneon1 = getdata("C:/Users/Jehanne/Desktop/Doctorat/FIRST-PL/Neon1/PL_Neon_dark.fits")

def picturethis(data, picnumber):
    plt.imshow(diff0, cmap='gray',norm=LogNorm())
    plt.colorbar()
    plt.show()

def diffdark(data, dark):
    return data-dark

limitehaute = 10000
limitebasse = 100

def format(data, limitebasse, limitehaute):
    format = data.copy()
    format[format> 10000] = 10000
    format[format < 100] = 0
    return format


def singleframe(data, n):
    return data[n]

image0 = singleframe(neon1,0)
dark0 = singleframe(darkneon1,0)

# Remplacer toutes les valeurs de image0_retouche qui sont > 200 par 1000

#image0_retouche[image0_retouche > 200] = 1000


# Vérifier que les deux images ont la même taille


def statsglobales(data):
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    print("nb of columunes : ", len(data))
    print("Valeur minimale :", min_val)
    print("Valeur maximale :", max_val)
    print("Valeur moyenne :", mean_val)

difference_image = diffdark(image0, dark0)
diff0 = format(difference_image, limitebasse, limitehaute)

para1 = 1300000

def identifywavelenght(data,para1):
    data_transposed = data.T
    #print(data_transposed)
    row_sums = data_transposed.sum(axis=1)
    #print(row_sums)
    #statsglobales(row_sums)


    indices = np.where(row_sums < para1)[0]

    # Convert indices to a list (optional)
    indices_list = indices.tolist()

    print("Indices de colonne où des valeurs sont détéctées :",indices_list, "\n")

    result_list = []
    for i in range(len(indices_list)):
        # Si c'est le premier élément ou si l'élément précédent n'est pas consécutif
        if i == 0 or indices_list[i] != indices_list[i - 1] + 1:
            result_list.append(indices_list[i])

    print("Indices de colonnes où des raies commencent :",result_list, "\n")
    print("Nombre de raies : ",len(result_list),"\n")

identifywavelenght(diff0, para1)

