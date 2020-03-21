'''
This script sets up the simple flux model for Cambridge Bay 2018 trials.
'''

import gasex
import numpy as np
import matplotlib.pyplot as plt



def simulate(A, H, Q, S, T, W, methane_bay, methane_river, niter):
    ''' Forward Euler implementation for modeling methane in Cambridge Bay '''
    seconds = niter * 24. * 60. * 60. #TODO: Choose the resolution of the simulation

    for s in seconds:
        #airsea flux
        k = gasex.airsea.kgas(methane_bay, gasex.airsea.fsa_pC(S, T, methane_bay))
        F = A*k

        #river transport
        R = Q*methane_river - Q*methane_bay

        ch4_delta = R - F
        methane_bay = methane_bay + ch4_delta


if __name__ == '__main__':
    #Create the parameters for the simulation
    Area = 4000000 #square meters
    height = 2.0 #meter
    river_flow = 0.1 #meters/second -- 0.1 on June 2, 39 on July 3
    sal = 0.01 #PSU
    temp = 5.0 #Celsius
    windspeed = 0.1 #meter/second
    ch4_bay = 300.0 #mol/m3
    ch4_river = 400.0 #mol/m3

    #Set the simulation time (in days)
    niter = 5

    #Run the simulation
    ch4_bay_progression = simulate(Area, height, river_flow, sal, temp, windspeed, ch4_bay, ch4_river, niter)