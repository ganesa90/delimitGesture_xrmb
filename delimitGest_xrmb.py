# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 01:59:41 2017

@author: ganesh
"""

import numpy as np
import os
import sys
import scipy.signal
import scipy.io

def get_gesture(tv, thrnons, thrnoff):
    # Mean and variance normalize the TV
    tv = (tv - np.mean(tv))/np.std(tv)    
    # Compute the absolute velocity 
    v = np.concatenate([np.diff(tv[0:2]), tv[2:]-tv[0:-2], np.diff(tv[-2:])], axis=0)/2
    act_v = v
    v = abs(v)    
    v = v - min(v)
    # Compute the filtered velocity
    [b, a] = scipy.signal.butter(3, 0.2, output='ba')
    fv = scipy.signal.filtfilt(b,a,v)			# Butterworth LP @ 10%
    
    # FInd the ninima on the filtered signal
    accl_dir = np.array(np.concatenate([np.diff(fv[0:2]), np.diff(fv)]) > 0, dtype=int)    
    vel_minima = np.where(np.diff(accl_dir) > 0)[0]
    minima_list = []
    if vel_minima[0] > 0:
        minima_list.append(0)
    minima_list+=list(vel_minima)
    if vel_minima[-1] < (tv.shape[0]-1):
        minima_list.append(tv.shape[0]-1)
    
    maxV = max(v)    
    onsThr = 0.1
    gest = {}
    gest['NONS'] = []
    gest['NOFF'] = []
    bin_gest = np.zeros(tv.shape[0])
    for ii, indx in enumerate(minima_list[1:-1]):
        ons_ind = ii
        MAXC = indx
        while ons_ind >= 0:
            GONS = minima_list[ons_ind]
            if abs(max(fv[GONS:MAXC])) - min(fv[GONS:MAXC]) > onsThr*maxV:
                break
            ons_ind = ons_ind - 1
#        if ons_ind < 0:
#            continue
        off_ind = ii+2
        while off_ind <= (len(minima_list)-1):
            GOFF = minima_list[off_ind]
            if abs(max(fv[MAXC:GOFF])) - min(fv[MAXC:GOFF]) > onsThr*maxV:
                break
            off_ind = off_ind + 1
#        if off_ind > (len(minima_list)-1):
#            continue
        
        PVEL1 = np.argmax(v[GONS:MAXC])
        PVEL1 = PVEL1 + GONS
        PVEL2 = np.argmax(v[MAXC:GOFF])
        PVEL2 = MAXC + PVEL2
        
        # Adjust MAXC using unfiltered velocity signal
        MAXC = np.argmin(v[PVEL1:PVEL2])
        MAXC = PVEL1 + MAXC
        
        # Check if MAXC is a peak point in TV. We will discard it
        if (tv[MAXC] > tv[PVEL1]) and (tv[MAXC] > tv[PVEL2]):
            continue
        # Check whether the dip in the TV is below zero
        if tv[MAXC] > 0:
            continue        
        # Set onsets using threshold criterion
        NONS_range = np.where((v[PVEL1:MAXC] - v[MAXC]) > thrnons*(v[PVEL1] - v[MAXC]))[0]
        NOFF_range = np.where((v[MAXC:PVEL2] - v[MAXC]) > thrnons*(v[PVEL2] - v[MAXC]))[0]
        
        if NONS_range.size == 0:
            NONS_st = 0
        else:
            NONS_st = NONS_range[-1]
        
        if NOFF_range.size == 0:
            NOFF_en = PVEL2 - MAXC
        else:
            NOFF_en = NOFF_range[0]
        
        NONS = PVEL1 + NONS_st
        NOFF = indx + NOFF_en
        bin_gest[NONS:NOFF] = 1.0
        
        gest['NONS'].append(NONS)
        gest['NOFF'].append(NOFF)
    
#    plt.plot(tv, label='Tract Variable', color='k')
#    plt.plot(bin_gest, label='Binary gesture signal', color='r')  
#    plt.plot(fv, label='Smoothed velocity', color='g')
#    minima_bin = np.zeros(tv.shape[0])
#    minima_bin[vel_minima] = 1.0
#    plt.stem(minima_bin, label='Velocity minima')
#    plt.legend()
    return gest, bin_gest, tv
    
if __name__ == "__main__":
    filelist = sys.argv[1]
    opdir = sys.argv[2]
    if not os.path.exists(opdir):
        os.makedirs(opdir)    
    thrnons = 0.7
    thrnoff = 0.7
    
    with open(filelist) as fp:
        flist = fp.read().splitlines()
    for fpath in flist:
        fnm = fpath.split('/')[-1].split('.')[0]
        opfnm = opdir+'/'+fnm+'.mat'
        matfile = scipy.io.loadmat(fpath)
        TV = matfile['tv_norm']        
        gest_bin = np.zeros((3, TV.shape[1]))
        
        # Compute gestures and store them in gest_bin
        tv = TV[0,:]   # Lip Aperture
        gest, bin_g, tv_norm = get_gesture(tv, thrnons, thrnoff)
        gest_bin[0,:] = bin_g
    
        tv = TV[3,:]   # TBCD
        gest, bin_g, tv_norm = get_gesture(tv, thrnons, thrnoff)   
        gest_bin[1,:] = bin_g
    
        tv = TV[5,:]   # TTCD
        gest, bin_g, tv_norm = get_gesture(tv, thrnons, thrnoff)
        gest_bin[2,:] = bin_g
        
        opmatfile = {}
        opmatfile['gest'] = gest_bin
        scipy.io.savemat(opfnm+'.mat', opmatfile)

