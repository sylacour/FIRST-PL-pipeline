import subprocess
import os
from glob import glob
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import runPL_library as runlib
from runPL_createPixelMap import run_createPixelMap
from runPL_preprocess import preprocess
from runPL_createWavelengthMap import runCreateWavelengthMap
from runPL_createWavelengthMap import runForStar
import numpy as np



def promptContinue(command="", yesInput=False):
    '''
    Ask the user if they'd like to continue the program
    '''
    if not yesInput:
        user_input = input(f"\n Execute the following command ? (y/n) : \n{command}\n")
    else : 
        return True

    if user_input.lower() in ['oui', 'o', 'yes', 'y']:
        return True
    print("Command skipped")
    return False




def NeonCalibration(whereFiles, wavelist, allYesInput =False):

    permission = promptContinue(command="Create pixel map", yesInput=allYesInput )
    if permission:
        run_createPixelMap(whereFiles, destination=whereFiles, pixel_min=100, pixel_max=1600, pixel_wide=3, output_channels=38, file_patterns=["*.fits"])

    permission = promptContinue(command="Run preprocess", yesInput=allYesInput )
    if permission:
        preprocess(folder = whereFiles,pixel_map_file = None)

    permission = promptContinue(command="Create wavelength map", yesInput=allYesInput)
    if permission:
        output = runCreateWavelengthMap(whereFiles+"preproc/", wavelist, saveWhere=whereFiles+"calibration_result/")
        np.savetxt("WavePolyBest.txt", output)
        print("Best polyfit output saved : "+os.path.abspath("WavePolyBest.txt"))
        saveTXT_wavelength_pixel(output, whereFiles+"calibration_result/", "pixels_to_wavelength.txt")

    return True

def saveTXT_wavelength_pixel(wavePoly, whereToSave, filename):
    bestFit = np.poly1d(wavePoly)
    pixels = np.arange(0, 5001) 
    wavelengths = bestFit(pixels)

    data = np.column_stack((pixels, wavelengths))
    np.savetxt(os.path.join(whereToSave,filename), data, fmt="%.6f", header="Pixels Wavelengths", comments="")
    



def getPoly_from_txt():
    file_name="WavePolyBest.txt"
    if os.path.exists(file_name):
        # File exists, read it
        WavePolyBest_loaded = np.loadtxt(file_name)
        print("Loaded coefficients : ", WavePolyBest_loaded)
        return WavePolyBest_loaded
    else:
    # File does not exist
        print(f"Error: File '{file_name}' does not exist.")
    return 
    


def reduct_star_data(whereFiles, allYesInput =False):
    permission = promptContinue(command="Create pixel map", yesInput=allYesInput )
    if permission:
        run_createPixelMap(whereFiles, destination=whereFiles, pixel_min=100, pixel_max=1600, pixel_wide=3, output_channels=38, file_patterns=["*.fits"])

    permission = promptContinue(command="Run preprocess", yesInput=allYesInput )
    if permission:
        filelist_pixelmap = runlib.clean_filelist({"DATA-CAT": 'PIXELMAP'}, runlib.get_filelist(os.path.join(whereFiles, "pixelmaps")))
        preprocess(filelist_pixelmap, files_by_dir={whereFiles:whereFiles})
    
    permission = promptContinue(command="Shift to wavelenght", yesInput=allYesInput )
    if permission:
        poly = getPoly_from_txt()
        runForStar(whereFiles+"preproc/", poly, whereFiles+"output/")
    
def reduce_many_star_data(parent_folder):
    '''
    
    
    '''
    fits_keywords = {'DATA-CAT': ['RAW'], 
                        'DATA-TYP': ['OBJECT']}
    filelist=[]
    for file in os.listdir(parent_folder):
        if file.endswith(".fits"):
            filelist.append(os.path.join(parent_folder, file))
    
    fits_keywords = {'DATA-CAT': ['RAW'], 
                        'DATA-TYP': ['OBJECT']}
    filelist_star = runlib.clean_filelist(fits_keywords, filelist)
    fits_keywords = {'DATA-CAT': ['RAW'], 
                        'DATA-TYP': ['DARK']}
    filelist_dark = runlib.clean_filelist(fits_keywords, filelist)

    for star in filelist_star:
        closest_dark_files = runlib.find_closest_dark(star, filelist_dark, filter_by_directory = True)
        folder_name = str(star)[:-5]+"/"
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(str(star)[:-5], exist_ok=True)
        name_star = os.path.basename(star)
        name_dark = os.path.basename(closest_dark_files)
        shutil.copy(star, folder_name+name_star)
        shutil.copy(closest_dark_files, folder_name+name_dark)
        print("\nReducing data for : "+str(star))
        reduct_star_data(folder_name, allYesInput=True)



if __name__ == "__main__":
    '''
    Neon or single star source folder must contain only two fits file : the file to analyze and its dark.
    Outputs (pixelmaps, preprocess, result) will be saved inside separate folder at the same place

    The full star folder can be filled directly with all fits files and their darks. 
    For each star found, the code will create a folder and copy both the star and its dark inside. 
    All results will be saved in this folder.
    
    '''
    neon1 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon1/"
    neon2 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon2/"
    neon3 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon3/"
    neon4 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon4/"
    lesautres = "/home/jsarrazin/Bureau/PLDATA/InitData/"
    onestar ="/home/jsarrazin/Bureau/PLDATA/InitData/im_cube_2024-08-18_11-01-23_altair/"
    nov1 = "/home/jsarrazin/Bureau/PLDATA/novembre/2024-11-21_13-48-32_science/"
    nov2 = "/home/jsarrazin/Bureau/PLDATA/novembre/2024-11-21_14-09-14_science/"
    nov3 = "/home/jsarrazin/Bureau/PLDATA/novembre/2024-11-21_14-36-09_science/"
    nov4 = "/home/jsarrazin/Bureau/PLDATA/novembre/2024-11-21_15-43-53_science/"
    nov5 = "/home/jsarrazin/Bureau/PLDATA/novembre/2024-11-21_16-03-39_science/"


    #Change whereFiles accordingly
    whereFiles = nov1

    # à la main avec neon1 :
    wavelist = "[748.9, 724.5, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 626.7]"

    user_input = input("Neon calibration or star spectra ? (neon/star)")
    if user_input.lower() =="star":
        user_input = input("Single star or full folder ? (single/folder)")
        if user_input.lower() =="single":
            reduct_star_data(whereFiles )
        elif user_input.lower() =="folder":
            reduce_many_star_data(whereFiles)
        else:
            print("input not recognized")
    elif user_input.lower() =="neon":
        NeonCalibration(whereFiles,wavelist,False)
        print("The polyfit txt is updated")
    else:
        print("input not recognized")