import subprocess

# Liste des commandes à exécuter, avec leurs arguments respectifs
commands = [
    "python runPL_compareCouplingMaps.py --cmap_size=25 *.fits",
    "python runSecondScript.py --arg1=value1 --arg2=value2",
    "python runThirdScript.py --arg=value *.fits"
]

execPython = " & C:/Users/Jehanne/AppData/Local/Programs/Python/Python312/python.exe"
execScript = "c:/Users/Jehanne/Desktop/Doctorat/git/FIRST-PL-pipeline/runPL_createPixelMap.py --pixel_min=100 --pixel_max=1600 --pixel_wide=2 --output_channels=38"
whereFiles = "C:/Users/Jehanne/Desktop/Doctorat/FIRST-PL/*.fit"

# Exécute chaque commande en séquence
for command in commands:
    try:
        print(f"Exécution de la commande : {command}")
        result = subprocess.run(command, shell=True, check=True)
        print(f"Commande terminée avec succès : {command}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la commande : {command}")
        print(f"Code de retour : {e.returncode}")
        break  # Stopper si une commande échoue
