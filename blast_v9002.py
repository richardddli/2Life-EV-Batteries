#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:40:11 2019

@author: saraborchers
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pandas as pd
import pdb

#Universal constants
T_ref = 298.15
F = 96485
U_ref = 0.08
R_ug = 8.314

def plot (x,y):
    fig,ax = plt.subplots()
    ax.plot(x,y)
    #ax0.set_ylim(0.6,1.6)
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))

fig,ax = plt.subplots()


###############################################################################
############ INTERPOLATION FUNCTIONS USING KANDLER'S LOOKUP TABLES ############
###############################################################################
    
def get_V_from_SOC(SOC):
    soc_index = np.arange(0, 1.1, 0.1)
    ocv_index = [3.0000, 3.4679, 3.5394, 3.5950, 3.6453, 3.6876, 3.7469, 3.8400, 3.9521, 4.0668, 4.1934]
    return np.interp(SOC, soc_index, ocv_index)

#  NOT BEING USED (crazy results)
def get_U_from_V (V):
    Vtest = [3,3.4679,3.5394,3.5950,3.6453,3.6876,3.7469,3.84,3.9521,4.0668,4.1934]
    Utest = [1.2868,.2420,.1818,.1488,.1297,.1230,.1181,.1061,.0925,.0876,.0859]
    Ut = []
    for x in V:
        U = np.interp(x,Vtest,Utest)
        Ut.append(U)
    return Ut

    
###############################################################################
####################### CALCULATING KEY MODEL PARAMETERS ######################
###############################################################################

def getQneg (Tt, N, DOD):
    c2_ref = 3.9193e-3
    beta_c2 = 4.54
    Ea_c2 = -48260
    c2 = c2_ref * np.exp(-Ea_c2 / R_ug * (1/Tt - 1/T_ref))  * np.power(DOD, beta_c2)
    
    c0_ref = 75.64
    Ea_c0 = 2224
    c0 = c0_ref * np.exp(-Ea_c0 / R_ug * (1/Tt - 1/T_ref))
    Q_neg = np.sqrt(np.power(c0,2) - 2 * c2 * c0 * N)    
    return Q_neg

def getCapacity(t, T, V, U, N, DOD, second_life=False):
    
    # find Q_pos
    d3 = 0.46
    d0_ref = 75.1
    Ea_d0_1 = 34300
    Ea_d0_2 = 74860
    d0 = d0_ref * np.exp(-Ea_d0_1 / R_ug * (1/T - 1/T_ref) - np.power(Ea_d0_2/R_ug, 2) * np.power(1/T - 1/T_ref, 2))
    if second_life:
        d0 *= 0.8
    Ah_dis = t / 0.1 * DOD * 75
    Q_pos = d0 + d3 * (1 - np.exp(-Ah_dis / 228))
    
    # find Q_Li
    b1_ref = 3.503e-3
    Ea_b1 = 35392
    alpha_b1 = 1
    gamma_b1 = 2.472
    beta_b1 = 2.157
    b0 = 1.07
    b3_ref = 2.805e-2
    Ea_b3 = 42800
    alpha_b3 = 0.0066
    tau_b3 = 5
    theta = 0.135
    b2_ref = 1.541e-5
    Ea_b2 = -42800
    U_ref = 0.08
    V_ref = 3.7
    DOD_max = DOD
    
    b1 = b1_ref * np.exp(-Ea_b1/R_ug * (1/T - 1/T_ref)) * np.exp(alpha_b1 * F/R_ug * 
                        (U/T - U_ref/T_ref)) * np.exp(gamma_b1 * np.power(DOD_max, beta_b1))
    b2 = b2_ref * np.exp(-Ea_b2/R_ug * (1/T - 1/T_ref))
    b3 = b3_ref * np.exp(-Ea_b3/R_ug * (1/T - 1/T_ref)) * np.exp(alpha_b3 * F/R_ug * 
                        (V/T - V_ref/T_ref)) * (1 + theta * DOD_max)
    Q_Li = d0 * (b0 - b1 * np.sqrt(t) - b2 * N - b3 * (1-np.exp(-t/tau_b3)))
    
    # Q is the minimum of these three Q contributions
    
    return Q_Li, Q_pos


def getR (t, T_rptt, Tt, Ut, Q_neg, DOD, second_life=False):
    #Calculate R0
    R_0ref = .001155
    E_aR0 = -28640
    
    R0 = R_0ref * np.exp((-E_aR0/R_ug)*((1/T_rptt)-(1/T_ref)))
    if second_life:
        R0 *= 1.25
    
    #Calculate a0
    a_01 = 0.442
    a_02 = -0.199
    e_aa01 = 28640
    e_aa02 = -46010
    
    a0_term1 = (-e_aa01/R_ug)*((1/T_rptt)-(1/T_ref))
    a0_term2 = (-e_aa02/R_ug)*((1/T_rptt)-(1/T_ref))
    a0 = (a_01 * np.exp(a0_term1)) + (a_02 * np.exp(a0_term2))
    
    #Calculate a1
    a_1ref = 0.0134
    E_aa1 = 36100
    alpha_a1 = -1
    gamma_a1 = 2.433
    beta_a1 = 1.870
    
    a1_term1 = (-E_aa1/R_ug)*((1/Tt)-(1/T_ref))
    a1_term2 = (alpha_a1*F/R_ug)*((Ut/Tt)-(U_ref/T_ref))
    a1_term3 = gamma_a1*DOD**beta_a1
    a1 = a_1ref * np.exp(a1_term1) * np.exp(a1_term2) * np.exp(a1_term3)
    
    #Calculate a2
    a_2ref = 46.05
    E_aa2 = -29360
    
    a2 = a_2ref * np.exp((-E_aa2/R_ug)*((1/Tt)-(1/T_ref)))
    
    #Calculate a3
    a_3ref = 0.145
    E_aa3 = -29360
    
    a3 = a_3ref * np.exp((-E_aa3/R_ug)*((1/Tt)-(1/T_ref)))
    
    #Calculate a4
    a_4ref = 0.0005357
    E_aa4 = 77470
    alpha_a4 = -1
    
    a4_term1 = (-E_aa4/R_ug)*((1/Tt)-(1/T_ref))
    a4_term2 = ((alpha_a4*F)/R_ug)*((Ut/Tt)-(U_ref/T_ref))
    a4 = a_4ref * np.exp(a4_term1) * np.exp(a4_term2)
    
    #Calculate R
    tau_a3 = 100
    R = 1000*(R0 * (a0 + a1* np.sqrt(t) + a2/Q_neg  + a4*t))
    # a3 break-in term not being used
    #- a3*(1-np.exp(-t/tau_a3))
    return R


###############################################################################
##################### HELPER FUNCTIONS FOR APPLICATIONS #######################
###############################################################################

# finds number of (partial) cycles in one day. for frequency regulation case
def count_N(df):
    one = False
    count = 0
    threshold = 0.98
    for i, row in df.iterrows():
        if not one:
            if row['frequency'] >= threshold:
                one = True
                count += 1
        if one:
            if row['frequency'] < threshold:
                one = False
    return count

# creates SOC curve for fast charging
def get_SOC_FC (DOD):
    Ah = 75
    start_SOC = .5 + DOD/2
    end_SOC = start_SOC-DOD
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
    
    # add idle time to make N=12/day
    SOC += [start_SOC]*(120-len(SOC))
    return SOC     

                
###############################################################################
################## HELPER FUNCTIONS FOR NREL TEST CASES #######################
###############################################################################

# Constants that change based on test case. 
# Returns: (days, N, DOD, rpt)
def get_constants(battery):
    if battery == 1:
        return 295, 3020, 0.8, 23
    if battery == 3:
        return 275, 2400, 1, 30
    if battery == 9:
        return 210, 2250, 0.8, 45

# recreates voltage profile from NREL test cases, with N = 10.4 cycles/day
def get_V_test(DOD, days):
    if DOD == 1:
        V_oc = [4.2, 3.95, 3.85, 3.75, 3.7, 3.65, 3.6, 3.57, 3.52, 3.4, 3]
    if DOD == 0.8:
        V_oc = [4.1, 3.92, 3.85, 3.76, 3.7, 3.66, 3.62, 3.6, 3.55, 3.5, 3.4]
    V_oc = V_oc + V_oc[-2:0:-1] + V_oc[0:1] + V_oc[0:1] + V_oc[0:1] 
    V_oc = np.tile(V_oc, int(days*240/20))
    V_oc = V_oc[:days*240]
    return V_oc

# (old) internal temperature lookup
def get_T_int_from_T_ext (rpt):
    T_ext = [ 23, 30, 45]
    T_int = [ 299, 304,325] 
    return np.interp(rpt, T_ext, T_int)


###############################################################################
############################# MAIN FUNCTIONS  #################################
###############################################################################

def run_NREL_test_case(battery, ax1=None, ax2=None):
    days, N, DOD, rpt = get_constants(battery)
    N *= DOD
    
    #create dataframe which will store all time-dependent variables
    df = pd.DataFrame(index = np.arange(0, days, 1/240))
    df['V'] = get_V_test(DOD, days)
    df['N'] = df.index * N / days
    df['T_rpt'] = np.ones(days*240)*(rpt+273.15)
    df['T'] = np.ones(days*240) * get_T_int_from_T_ext(rpt)
    df['U_for_R'] = np.ones(days*240) * 0.123
    df['U_for_Q'] = np.ones(days*240) * 0.03
    
    Q_initial = 75
    DOD_store = DOD
    for i, rows in df.groupby(df.index // 1):
        t = rows.index
        df.loc[t, 'DOD'] = DOD_store
        df.loc[t, 'Q_neg'] = getQneg(rows['T'], rows['N'], df.loc[t, 'DOD'])
        df.loc[t, 'R'] = getR(t, rows['T_rpt'], rows['T'], rows['U_for_R'], df.loc[t, 'Q_neg'], df.loc[t, 'DOD'], )
        df.loc[t, 'Q_Li'], df.loc[t,'Q_pos'] = getCapacity(t, rows['T'], rows['V'], rows['U_for_Q'], rows['N'], df.loc[t, 'DOD'], second_life=False)
        df.loc[t, 'Q'] = df.loc[t,['Q_Li', 'Q_pos', 'Q_neg']].min(axis=1)
        DOD_store = DOD * Q_initial / np.mean(df.loc[t, 'Q'])
        if DOD >= 1:
            break
    df = df[df['DOD'].notnull()]
    # if an axis is passed in, plot R
    if ax1:
        ax1.plot(df.index, df['R'], label="(%i) %iC, %.1f DOD" %(battery, rpt, DOD))
    
    if ax2:
        ax2.plot(df.index, df['Q'], label="(%i) %iC, %.1f DOD" %(battery, rpt, DOD))
    
    return df, ax1, ax2

def run_freq_regulation(filename, DOD, ax1=None, ax2=None):
    days = 250
    DOD = DOD # determined to be best for battery lifetime
    rpt = 23   
    freq_data = pd.read_csv(filename, usecols=['frequency'])
    
    #directly convert frequency to SOC
    max_freq = freq_data.describe().loc['max','frequency']
    min_freq = freq_data.describe().loc['min','frequency']
    df = freq_data
    df['SOC'] = (df['frequency'] - min_freq) / (max_freq - min_freq) * DOD + 0.5-DOD/2
    N = count_N(df) * days * DOD
    
    #find V from SOC using lookup tables (interpolation)
    df['V'] = df['SOC'].apply(lambda x: get_V_from_SOC(x))
    
    #copy 1 day of data over 250 days
    df = pd.concat([freq_data.iloc[:-1]]*days)
    
    df['t'] = np.arange(0,days, 1/24/60/30)
    df['yr'] = df['t']/365
    # sets index (left most column) of df to time
    df.set_index('t', drop=True, inplace=True)
    
    df['U_for_R'] = 0.123
    df['U_for_Q'] = 0.03
    df['N'] = df.index * N / days
    df['T'] = get_T_int_from_T_ext(rpt)
    df['T_rpt'] = rpt+273.15
    df['Q_neg'] = getQneg(df['T'], df['N'], DOD)
    df['R'] = getR(df.index, df['T_rpt'], df['T'], df['U_for_R'], df['Q_neg'], DOD, second_life=True)
    df['Q_Li'], df['Q_pos'] = getCapacity(df.index, df['T'], df['V'], df['U_for_Q'], df['N'], DOD, second_life=True)
    df['Q'] = df[['Q_Li', 'Q_pos', 'Q_neg']].min(axis=1)

#    if ax1:
#        ax1.plot(df.index, df['Q'], label='DOD=%.2f' %DOD)
#    if ax2:
#        ax2.plot(df.index, df['R'], label='DOD=%.2f' %DOD)
    
    return df, ax1, ax2

def run_fast_charge(DOD, ax1=None, ax2=None):
    days = 14*365
    rpt = 23
    N = 12  * days *  DOD
    
    df = pd.DataFrame()
    df['SOC'] = get_SOC_FC(DOD)
    df['V'] = df['SOC'].apply(lambda x: get_V_from_SOC(x))
    df = pd.concat([df] * int(days*24/2))
    df['t'] = np.arange(0, days, 1/24/60)[:-1]
    df.set_index('t', drop=True, inplace=True)
    
    df['U_for_R'] = 0.123
    df['U_for_Q'] = 0.03
    df['N'] = df.index * N / days
    df['T'] = get_T_int_from_T_ext(rpt)
    df['T_rpt'] = rpt+273.15
    
    Q_initial = 75
    DOD_store = DOD
    for i, rows in df.groupby(df.index // 4):
        t = rows.index
        df.loc[t, 'DOD'] = DOD_store
        df.loc[t, 'Q_neg'] = getQneg(rows['T'], rows['N'], df.loc[t, 'DOD'])
        df.loc[t, 'R'] = getR(t, rows['T_rpt'], rows['T'], rows['U_for_R'], df.loc[t, 'Q_neg'], df.loc[t, 'DOD'], second_life=True)
        df.loc[t, 'Q_Li'], df.loc[t,'Q_pos'] = getCapacity(t, rows['T'], rows['V'], rows['U_for_Q'], rows['N'], df.loc[t, 'DOD'], second_life=True)
        df.loc[t, 'Q'] = df.loc[t,['Q_Li', 'Q_pos', 'Q_neg']].min(axis=1)
        DOD_store = DOD * Q_initial / np.mean(df.loc[t, 'Q'])
        if DOD_store >= 1:
            break
    
    df = df[df['DOD'].notnull()]
        
#    df['Q_neg'] = getQneg(df['T'], df['N'], DOD)
#    df['R'] = getR(df.index, df['T_rpt'], df['T'], df['U_for_R'], df['Q_neg'], DOD)
#    df = getCapacity(df, df['N'], DOD, second_life=True)
    
    # if an axis is passed in, plot R and Q
    if ax1:
        ax1.plot(df.index, df['Q'], label="%.1f DOD" %(DOD))
    if ax2:
        ax2.plot(df.index, df['R'], label="%.1f DOD" %(DOD))
    return df, ax1, ax2

###############################################################################
######################### ACTUALLY RUN SHIT HERE  #############################
###############################################################################


#RUN NREL test cases
#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
## change to [1] if you only want to see one battery:
#for battery in [9]:
#    df, ax1, ax2 = run_NREL_test_case(battery, ax1, ax2)
#    
#ax1.set_xlabel('days')
#ax1.set_ylabel('resistance (ohms)')
#ax1.set_title('resistance growth')
#ax1.legend()
#fig1
#
#ax2.set_xlabel('days')
#ax2.set_ylabel('capacity (Ah)')
#ax2.set_title('capacity fade')
#ax2.legend()
#fig2


## RUN FREQUENCY REGULATION
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# change if you only one want DOD:
for DOD in [0.3]:
    df, ax1, ax2 = run_freq_regulation('rto-regulation-signal-for-july-19-20.csv', DOD, ax1, ax2)
ax1.set_xlabel('days')
ax1.set_ylabel('capacity (Ah)')
ax1.set_title('frequency regulation: capacity fade')
ax1.legend()
fig1

ax2.set_xlabel('days')
ax2.set_ylabel('resistance (ohms)')
ax2.set_title('frequency regulation: resistance growth')
ax2.legend()
fig2


## RUN FAST CHARGING for 3 DODs
#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
## change if you only one want DOD:
#for DOD in [0.7]:# 0.5, 0.8]:
#    df, ax1, ax2 = run_fast_charge(DOD, ax1, ax2)
#ax1.set_xlabel('days')
#ax1.set_ylabel('capacity (Ah)')
#ax1.set_title('frequency regulation: capacity fade')
#ax1.legend(loc='upper left')
#fig1
#
#ax2.set_xlabel('days')
#ax2.set_ylabel('resistance (ohms)')
#ax2.set_title('frequency regulation: resistance growth')
#ax2.legend(loc='upper left')
#fig2

#0.3:
#2000.000694    1.844417
#4000.000694    2.391724
#0.0002736535

##.1:
#2.5, 14
#62, 56.5
#
##.3:
#1, 12
#62, 46.5
#
##.5:
#220 days
#
#0.7
#130 days

