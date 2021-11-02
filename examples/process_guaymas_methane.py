from gasex import sol
import numpy
import matplotlib.pyplot as plt


def convert_CH4(x, eff=0.15, peq=495.):
    ''' Method to convert the raw ppm measurements from the GGA to compensated
    uatm units '''
    gppm = 1.834  # hardcoded constant
    ui = peq * gppm / 1000.
    return (x * peq / 1000. - ui) / eff + ui


def apply_efficiency(x, eff=0.15, gppm=1.86, peq=495.):
    '''Method for applying the extraction efficiency'''
    return ((x-gppm)/eff + gppm) * (peq/1000.)


CTD_meth = [3.5518, 3.1656, 3.0464, 3.2566, 2.9212, 2.3416, 2.1555, 2.2008]
CTD_temp = [9.3, 9.5, 10, 10.5, 10.8, 11.8, 13.4, 13.3]
CTD_salt = [33.8, 33.8, 33.7, 33.7, 33.6, 33.3, 33.3, 33.3]
CTD_depth = [45, 30, 25, 20, 12, 8, 5, 2]

nm = []

CTD_meth_uatm = [apply_efficiency(x, eff=0.0509)*0.000001 for x in CTD_meth]

for m, t, s, d in zip(CTD_meth_uatm, CTD_temp, CTD_salt, CTD_depth):
    nm.append(sol.sol_SP_pt(s, t, gas='CH4', p_dry=m, units='mM')/0.000001)

print(nm)

temp = convert_CH4(20, eff=1.0)/0.000001
print(sol.sol_SP_pt(10, 17, gas='CH4', p_dry=temp, units='mM') * 0.000001)
