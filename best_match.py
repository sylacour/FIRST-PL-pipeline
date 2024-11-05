# Set up matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
# C:\Users\Jehanne\Desktop\Doctorat\FIRST-PL\Neon1\PL_Neon.fits
from astropy.utils.data import download_file
import numpy as np

#wavelength_list = [753.6, 17] 
#detectedWavePeaks = np.linspace(11, 1000, 4)

#INPUT
#we need to give a wavelenght_list :
##wavelength_list = [753.6, 17] 
# and a detected peak list : 
## detectedWavePeaks = [3, 7, 17, 190, 300]

# The output must be output = [17, 300]
#Thats to say : 
# Detect ALL best peaks among all of those detected



def fun_ratio_ref(wavelength_list):
    # returns the ratio of all of the value by the minimum value of the list.
    # ex : [2,4,6,18] returns [1,2,3,9]
    wavelength_list=distance_from_first_value(wavelength_list)[0]
    ratio_wavelength =[]
    min_wavelength = min(wavelength_list)
    for wavelength in wavelength_list:
        ratio_wavelength.append(wavelength/min_wavelength)
    return ratio_wavelength

#print(fun_ratio_ref([2,4,6,18]))

def fun_ratio_detected(detectedWavePeaks):
    # returns a list of all possible ratios in the detectedWavePeaks.
    # ex : [2,4,6,18] returns :
    # [[1,2,3,9],
    # [0.5, 1, 1.5, 4.5],
    # [0.333, 0.666, 1, 3],
    # [0.111, 0.222, 0.333, 1]]
    firsts =[]
    all_ratio_peaks =[]
    for min_peak in detectedWavePeaks:
        min_ratio_peaks =[]
        a,b = distance_from_first_value(detectedWavePeaks)
        distance = a
        firsts.append(b)
        for peak in distance:
            min_ratio_peaks.append(peak/min_peak)
        all_ratio_peaks.append(min_ratio_peaks)
    return all_ratio_peaks, firsts

#print(fun_ratio_detected([2,4,6,18]))

def distance_from_first_value(datalist):
    #input is a list of number 
    #output returns the distance of each number to the first value
    # [2, 4, 8, 18] -> [2, 6, 16]
    firstvalue = datalist[0]
    distanceList = []
    for k in range(1,len(datalist)):
        distanceList.append(datalist[k]-firstvalue)
    return distanceList, firstvalue

#print(distance_from_first_value([2,4,8,18]))

def inner_list_candidate_answer(inner_list, ratio_wavelength):
    # for a single ratio list, returns the closest values to the ratio_wavelenght list
    # ex : if ratio_wavelength = [1,2,3,9] 
    # and inner_list = [0.111, 0.333, 1.1, 1.8, 2, 4, 10, 17, 100]
    # it will return [0.111, 2, 2, 10]
    # and corresponding list of index [0, 4 , 4, 6]
    all_distances = []
    valid_values =[]
    index_kept = []
    for reference_value in ratio_wavelength:
        closest_one = float('inf')  # Initialize with a large number
        valid_value = float(0)
        for value in inner_list : 
            suggestion = abs(reference_value-value)
            if suggestion<closest_one:
                closest_one=suggestion
                valid_value = value
        all_distances.append(closest_one)
        valid_values.append(0.0)
        valid_values[-1] = valid_value
        index_kept.append(inner_list.index(valid_value))
    return valid_values, index_kept

#print(inner_list_candidate_answer([0.111, 0.333, 1.1, 1.8, 2, 4, 10, 17, 100], [1,2,3,9] ))

def generate_best_candidates(all_ratio_peaks, ratioWavelength) : 
    # returns the closests candidates values of all calculated ratio peaks :
    # ex : if ratioWavelength = [1,2]
    # and all_ratio_peaks = [[1,2,3], [0.5, 1, 1.5], [0.333, 0.666, 1]]
    # it will return : [[1,2], [1, 1.5], [1, 1]]
    closest_candidates =[]
    corresponding_index =[]
    #print("We want to match the ratio : ", ratioWavelength)
    for ratioList in all_ratio_peaks:
        a,b = inner_list_candidate_answer(ratioList, ratioWavelength)
        closest_candidates.append(a)
        #print("Ratio list :", ratioList)
        #print("Best ratio for each ratio list : ",a)
        corresponding_index.append(b)
    #print(closest_candidates)
    return closest_candidates, corresponding_index

#print(generate_best_candidates([[1,2,3], [0.5, 1, 1.5], [0.333, 0.666, 1]], [1,2]))

def rating_candidate(candidate, ratio_wavelength):
    #print(candidate)
    # rates an inner ratio list based on how close to the ratiolist it is
    # ex : if ratio_wavelength = [1,2]
    # candidate [1,2] will return 0
    # candidate [1, 1.5] will return 0.5
    # best ratings will be closer to 0
    res = 0
    for k in range(len(ratio_wavelength)):
        res = res + abs(ratio_wavelength[k]-candidate[k])
    return res

#print(rating_candidates([1,2], [1,2]))
#print(rating_candidates([1,1.5], [1,2]))

def best_candidate(closest_candidates, ratioWavelength):
    # inputs all best candidates list (ex : [[1,2], [1, 1.5], [1, 1]])
    # and ratioWavelenght (ex : [1,2])
    # it will return the index of the best candidate list (here, #0)
    all_results =[]
    for candidate in closest_candidates:
        all_results.append(rating_candidate(candidate, ratioWavelength))
    actual_best = np.argmin(all_results)

    return actual_best

#print(best_candidate([[1,2], [1, 1.5], [1, 1]], [1,2]))

#weve lost which ratio equals to what actual value

#GENERATION

def BESTMATCH(wavelength_list, detectedWavePeaks):
    wavelength_list = sorted(wavelength_list)
    ratioWavelength = fun_ratio_ref(wavelength_list)
    #print(ratioWavelength)
    all_ratio_peaks, firsts = fun_ratio_detected(detectedWavePeaks)
    #print(all_ratio_peaks)
    a,b = generate_best_candidates(all_ratio_peaks, ratioWavelength)
    candidates = a
    candidates_actual_index = b
    resultat =best_candidate(candidates, ratioWavelength)
    #resultat is the index of the best candidate
    correct_inner_list = candidates[resultat]

    actual_values =[detectedWavePeaks[k] for k in candidates_actual_index[resultat]]
    actual_values.insert(0,firsts[resultat])

    return actual_values


wavelength_list = [17, 150, 753] 
# and a detected peak list : 
detectedWavePeaks = [3, 7, 17, 190, 300]
#print(BESTMATCH(wavelength_list, detectedWavePeaks))
# The output must be output = [17, 300]
#on perd la premiere valeur prise

#after some test :
wavelength_list = [753.6, 748.9, 743.9, 724.5, 717.4, 703.2, 693.0, 671.7, 667.8, 659.9, 653.3, 650.7, 640.2, 638.2, 633.4, 630.5, 626.7, 621.7, 616.4]

all_peaks = [48, 139, 147, 168, 178, 193, 210, 243, 297, 309, 407, 435, 505, 532,
                611, 619, 627, 645, 653, 711, 739, 747, 804, 920, 956, 994, 1071, 1136,
                1162, 1264, 1283, 1330, 1359, 1396, 1444, 1496]

regular_fit = [48, 147, 168, 178, 193, 243, 435, 645, 747, 956, 994, 1071, 1136, 1162,
                1264, 1283, 1330, 1396, 1496]

best_match_res = [48, 48, 139, 309, 407, 532, 653, 920, 956, 994, 1071, 1136, 1264, 1264, 1330, 1359, 1396, 1444, 1444]

def transposition_wave_to_pixel(wavesource, detectedsource):
    wavesource.sort()
    ratio_distance_wave = fun_ratio_ref(wavesource)
    print(ratio_distance_wave)
    init_distance = detectedsource[1]-detectedsource[0] #non valable quand 0 et 1 ont la meme valeur

    transposed = [detectedsource[0]]
    for k in range(0, len(wavesource)-1):
        transposed.append(ratio_distance_wave[k]*init_distance)
    return transposed


def matching_fig(wavelist, peaks): 
    fig, ax=plt.subplots(2, sharex=True)

    # Create a plot with vertical lines at each detected wave peak
    ax[0].vlines(wavelist, ymin=0, ymax=1, color='red', linestyle='--')
    ax[1].vlines(peaks, ymin=0, ymax=1, color='blue', linestyle='--')


    plt.show()
    
#print(BESTMATCH(wavelength_list, all_peaks))

#trans = transposition_wave_to_pixel(wavelength_list, all_peaks)
#print(matching_fig(trans, all_peaks))
