import numpy as np
import pandas as pd
import sys
import dtw
import sspd
import edr
import erp
import lcss
import sowd
import basic_euclidean as euclidean
import frechet
from L_anonymity import anonymity


def trajectory_similirity_compute(traj_A, traj_B):
    ######## Sequence-only and Discrete ########
    # x, y, zero, h, time : 3
    # ED - Euclidean distance calc
    edResult, edCalcTime= euclidean.ed_traj(traj_A, traj_B)
    print ("ED Distance: " + str(edResult) + " calcTime : " + str(edCalcTime))

    # DTW - Dynamic Time Warping calc
    dtwResult, dtwCalcTime = dtw.e_dtw(traj_A, traj_B)
    print ("DTW Distance: " + str(dtwResult) + " calcTime : " + str(dtwCalcTime))
    # ERP  - Edit distance with Real Penalty
    # g (gap) : parameter defalut value 0
    erpResult, erpCalcTime = erp.e_erp(traj_A, traj_B, 0)
    print ("ERP Distance: " + str(erpResult) + " calcTime : " + str(erpCalcTime))
    # EDR - Edit Distance on Real Sequence
    # esp : subcost(p1, p2) = 0 , d(p1, p2) ≤ ε 
    #                         1, otherwise
    edrResult, edrCalcTime = edr.e_edr(traj_A, traj_B, 0.1)
    print ("EDR Distance: " + str(edrResult) + " calcTime : " + str(edrCalcTime))
    # LCSS - Longuest Common Sebsequence dtwCalcTime
    # esp : subcost(p1, p2) = 0 , d(p1, p2) ≤ ε 
    #                         1, otherwise
    lcssResult, lcssCalcTime = lcss.e_lcss(traj_A, traj_B, 0.1)
    print ("LCSS Distance: " + str(lcssResult) + " calcTime : " + str(lcssCalcTime))
    # SSPD - Symmetric Segment-Path Distance calc
    sspdResult, sspdCalcTime = sspd.e_sspd(traj_A, traj_B)
    print ("SSPD Distance: " + str(sspdResult) + " calcTime : " + str(sspdCalcTime))
    ######## Sequence-only and Continuous ########
    # OWD - One-Way Distance calc
    sowdResult, sowdCalcTime = sowd.sowd_grid(traj_A, traj_B)
    print ("OWD Distance: " + str(sowdResult) + " calcTime : " + str(sowdCalcTime))
    ######## Spatial-temporal and Discrete ########

    ######## Spatial-temporal and Continuous ########
    # Frechet dtwCalcTime
    frechetResult, frechetCalcTime = frechet.frechet(traj_A, traj_B)
    print ("Frechet Distance: " + str(frechetResult) + " calcTime : " + str(frechetCalcTime))
    df = pd.DataFrame([
        ['ED',edResult, edCalcTime],['DTW',dtwResult, dtwCalcTime]
        ,['ERP', erpResult, erpCalcTime], ['EDR', edrResult, edrCalcTime]
        ,['LCSS', lcssResult, lcssCalcTime],['SSPD', sspdResult, sspdCalcTime]
        ,['OWD', sowdResult, sowdCalcTime],['Frechet', frechetResult, frechetCalcTime]]
        ,columns=['Name','Distance', 'Time'])
    return df

def getArrays(filename, skip_header):
    traj_list = np.genfromtxt(filename, delimiter=',', skip_header=skip_header)
    return np.array(traj_list)


def main():
    if len(sys.argv) < 3:
        print('You must input two filename to compare!')
        sys.exit(1)  # abort because of error

    filename1 = str(sys.argv[1])
    filename2 = str(sys.argv[2])

    ##### non-Anonymized calculation ########
    traj_A = getArrays(filename1, 7)
    traj_B = getArrays(filename2, 7)

    selection = np.array([True, True, False, False, False, False, False])
    traj_A = traj_A[:, selection]
    traj_B = traj_B[:, selection]
    result = trajectory_similirity_compute(traj_A, traj_B)
    result.to_csv('result.csv', index=True, header=True)
    ##### Anonymized calculation ########
    k_file1 = anonymity(filename1)
    k_file2 = anonymity(filename2)

    k_traj_A = getArrays(k_file1, 0)
    k_traj_B = getArrays(k_file2, 0)
    selection = np.array([True, True, False, False])
    k_traj_A = k_traj_A[:, selection]
    k_traj_B = k_traj_B[:, selection]
    result = trajectory_similirity_compute(k_traj_A, k_traj_B)
    result.to_csv('result_any.csv', index=True, header=True)

main()








