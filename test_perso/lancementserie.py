import subprocess
import os
from glob import glob
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import runPL_library as runlib
from runPL_createPixelMap_clean import run_createPixelMap
from runPL_preprocess_clean import run_preprocess
from runPL_createWavelengthMap_clean import runCreateWavelengthMap

neon1 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon1/*fits"
neon2 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon2/*fits"
neon3 = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon3/*fits"
lesautres = "/home/jsarrazin/Bureau/PLDATA/InitData/*fits"

whereFiles = neon2

# à la main avec neon1 :
wavelist = "[748.9, 724.5, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 626.7]"

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




def RunProg2(whereFiles=whereFiles, allYesInput =False):

    permission = promptContinue(command="Create pixel map", yesInput=allYesInput )
    if permission:
        run_createPixelMap(whereFiles, destination=whereFiles[:-5], pixel_min=100, pixel_max=1600, pixel_wide=3, output_channels=38, file_patterns=["*.fits"])

    permission = promptContinue(command="Run preprocess", yesInput=allYesInput )
    if permission:
        run_preprocess(folder = whereFiles,pixel_map_file = None)

    permission = promptContinue(command="Create wavelength map", yesInput=allYesInput)
    if permission:
        runCreateWavelengthMap(whereFiles+"preproc/", wavelist, saveWhere=whereFiles+"calibration_result/")

    return True


if __name__ == "__main__":
    RunProg2(whereFiles[:-5])
