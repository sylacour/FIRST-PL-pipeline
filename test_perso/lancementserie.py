import subprocess
import os
from glob import glob
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import runPL_library as runlib


# Liste des commandes à exécuter, avec leurs arguments respectifs
execPython = "/home/jsarrazin/anaconda3/envs/PLdev/bin/python "
execScript = "/home/jsarrazin/Bureau/GIT/REPERTOIRES/FIRST-PL-pipeline/runPL_createPixelMap.py --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --output_channels=38 *.fits~ "
whereFiles = "/home/jsarrazin/Bureau/PLDATA/InitData/Neon1/*fits"

pixelpath = "/home/jsarrazin/Bureau/GIT/REPERTOIRES/FIRST-PL-pipeline/output/pixel/"
pattern = os.path.join(pixelpath, "*.fits")
fits_files = glob(pattern)
if not fits_files:  # This checks if the list is empty
    most_recent_file=""
else:
    most_recent_file = max(fits_files, key=os.path.getmtime)

execpreprocess = "/home/jsarrazin/Bureau/GIT/REPERTOIRES/FIRST-PL-pipeline/runPL_preprocess.py --pixel_map="
pixelmappath = pixelpath+most_recent_file


execwave = "/home/jsarrazin/Bureau/GIT/REPERTOIRES/FIRST-PL-pipeline/runPL_createWavelengthMap.py " 
wavelist = "--wave_list=\"[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]\" *.fits "
#wave_list_string_default = "[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]"

#wavelist = "--wave_list=\"[753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693, 671.7, 667.8, 659.9]\" *.fits "

#wavelist = "--wave_list=\"[7245, 7173, 7032, 6929, 6598, 6402, 6382, 6334, 6266, 6217, 6163]\" *.fits "


preprocpath = whereFiles[:-5]+"preproc/*fits"
print(preprocpath)

execcopywave = "/home/jsarrazin/Bureau/GIT/REPERTOIRES/FIRST-PL-pipeline/createWave_copy.py "
generalpath = os.path.dirname(whereFiles)+"/*fits"
# Exécute chaque commande en séquence


user_input = input(f"\n N pour Neon, S pour series de fichiers :")
Neon = (user_input.lower()=="n")




def RunProgram(whereFiles=whereFiles,updateAsWeGo=True, allYesInput =False):
    preprocpath = whereFiles[:-5]+"preproc/*fits"
    command = [
    ["createPixelMap",execPython+execScript+whereFiles],
    ["runPL_preprocess",execPython+execpreprocess+pixelmappath+whereFiles],
    ["createWavelenghtMap",execPython+execcopywave+wavelist+preprocpath]
    ]
    commands = [
        ["createPixelMap",execPython+execScript+whereFiles], 
        ["runPL_preprocess",execPython+execpreprocess+pixelmappath+whereFiles],
        ["createWavelenghtMap",execPython+execwave+wavelist+preprocpath]
    ]
    
# Exécute chaque commande en séquence
    for command in commands:
        if updateAsWeGo==True:
            generalpath = preprocpath = os.path.dirname(whereFiles)+"/*fits"
            preprocpath = os.path.dirname(whereFiles)+"/preproc/*fits"
            if command[0]=="runPL_preprocess":
                pixelpath = "/home/jsarrazin/Bureau/GIT/REPERTOIRES/FIRST-PL-pipeline/output/pixel/"
                pattern = os.path.join(pixelpath, "*.fits")
                fits_files = glob(pattern)
                if not fits_files:  # This checks if the list is empty
                    most_recent_file=""
                else:
                    most_recent_file = max(fits_files, key=os.path.getmtime)+" "
                command[1] = execPython+execpreprocess+most_recent_file +whereFiles
            preprocpath = os.path.dirname(whereFiles)+"/preproc/*fits"
        # Demande de confirmation à l'utilisateur
        if not allYesInput:
            user_input = input(f"\n Voulez-vous exécuter la commande suivante ? (y/n):\n{command[0]}\n")
        else : 
            user_input="y"
        print("########### Running : "+command[0])
        print(preprocpath)
        if user_input.lower() in ['oui', 'o', 'yes', 'y']:
            try:
                result = subprocess.run(command[1], shell=True, check=True)
            except subprocess.CalledProcessError as e:
                break  # Stopper si une commande échoue
        else:
            print("skipped")
    

safeStop = 0
if Neon:
    RunProgram()
else:
    fits_keywords = {'DATA-CAT': ['RAW'],
                     'DATA-TYP': ['OBJECT']}
    filelist = []
    source = whereFiles[:-5]
    # Use the function to clean the filelist
    for file in os.listdir(source):
        if file.endswith(".fits"):
            filelist.append(source+file)

    filelist_cleaned = runlib.clean_filelist(fits_keywords, filelist)
    for k in range(0,len(filelist_cleaned)):
        filelist_cleaned[k]=os.path.basename(filelist_cleaned[k])

    dark_fits_keywords = {'DATA-CAT': ['RAW'],
                        'DATA-TYP': ['DARK']}
    dark_filelist_cleaned = runlib.clean_filelist(dark_fits_keywords, filelist)
    #for k in range(0,len(dark_filelist_cleaned)):
    #    dark_filelist_cleaned[k]=os.path.basename(dark_filelist_cleaned[k])

    for file in filelist_cleaned:
        safeStop=safeStop+1
        if safeStop > 1 : 
            raise SystemExit("safe stop")
        print("\n")
        print(file)
        foldername = file[:-5]
        output_dir = os.path.join(source, foldername)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(os.path.join(source, file), output_dir)
        linkedDark = runlib.find_closest_dark(source + file, dark_filelist_cleaned)
        shutil.copy(os.path.join(source, linkedDark), output_dir)
        whereFiles = os.path.join(output_dir, file)
        RunProgram(whereFiles)