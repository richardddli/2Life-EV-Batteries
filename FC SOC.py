#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:05:55 2019

@author: saraborchers
"""

def get_SOC_FC (DOD):
    if DOD == None:
        DOD = 0.62    

    Ah = 75
    start_SOC = .8
    end_SOC = start_SOC*(1-DOD)
    start_Ah = start_SOC * Ah
    end_Ah = end_SOC * Ah
    discharge_Crate = 3
    charge_Crate = 0.8

    discharge_minutes = (((start_Ah-end_Ah)*(1/discharge_Crate))/Ah)*60
    charge_minutes = (((start_Ah-end_Ah)*(1/charge_Crate))/Ah)*60
    discharge_slope = (end_SOC-start_SOC)/discharge_minutes
    charge_slope = (start_SOC-end_SOC)/charge_minutes
    SOC = []
    
    time = []
    for x in range (int(discharge_minutes+charge_minutes)):
        time.append(x)
    
    for x in range(int(discharge_minutes)):
        SOC.append(start_SOC + (time[x]*discharge_slope))

    for x in range(int(charge_minutes)):
        SOC.append(end_SOC + (time[x]*charge_slope))
        
    #correct for rounding errors
    SOC[-1]=start_SOC  
    return SOC        

    
SOC = get_SOC_FC(None)
