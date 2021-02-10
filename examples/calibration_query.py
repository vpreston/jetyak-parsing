#!/usr/env/python3

'''
Creates a probabilistic sampling strategy for bounding the error in flux calculations from data.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

# base libraries
import numpy as np
import scipy as sp
from parseyak import jetyak
from gasex import airsea
from gasex import sol
from gasex.diff import schmidt

# visualization functionality
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import LogFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def apply_efficiency(x, eff=0.15, gppm=1.86, peq=495.):
    '''Method for applying the extraction efficiency of the DGEU to
       compute compensated uatm from raw ppm values of the GGA'''
    return ((x-gppm)/eff + gppm) * (peq/1000.)

def importance_sampler(data, nsamps):
    ''' generates samples from an unknown distribution from data '''
    #get cdf of data
    data = data[~np.isnan(data)]
    hx, hy, _ = plt.hist(data, bins=100, density=1)
    dx = hy[1] - hy[0]
    F1 = np.cumsum(hx)*dx
    #uniformly sample from 0 to 1
    n = np.random.uniform(0, 1, nsamps)
    #query the cdf to get the potential generating value
    samps = np.interp(n, F1, hy[1:])
    plt.close()
    return samps

def run_ppm_nM_simulation(df, nsamps):
    ''' for a given dataset, pulls samples relevant for simulating the bounds on ppm to nM conversion '''
    pt = importance_sampler(df['Temperature'].values, nsamps) #sample temp
    SP = importance_sampler(df['Salinity'].values, nsamps) #sample salt
    wind = np.random.normal(10, 2, nsamps) #sample wind
    eff = np.random.normal(0.05, 0.01, nsamps) #sample extraction efficiency
    ch4 = np.linspace(2.0, 8.0, nsamps) #set methane values to analyze

    means = []
    stdevs = []
    med = []
    q1 = []
    q3 = []
    
    for m in ch4:
        measures = m * np.ones(eff.size) + np.random.normal(0, 0.0035, nsamps)
        gas_sig = apply_efficiency(measures, eff=eff)/0.000001 #gas sig as uatm
        gas_sig = sol.sol_SP_pt(SP, pt, gas='CH4', p_dry=gas_sig, units='mM') * 0.000001
        means.append(np.nanmean(gas_sig))
        stdevs.append(np.nanstd(gas_sig))
        med.append(np.nanmedian(gas_sig))
        q1.append(np.percentile(gas_sig, 25))
        q3.append(np.percentile(gas_sig, 75))

    means = np.asarray(means)
    stdevs = np.asarray(stdevs)

    return ch4, means, stdevs, med, q1, q3

def run_flux_simulation(df, nsamps):
    ''' for a given dataset, pulls samples relevant for simulation flux calculation '''
    ch4 = importance_sampler(df['CH4_ppm'].values, nsamps) + np.random.normal(0, 0.0035, nsamps) #sample ch4
    pt = importance_sampler(df['Temperature'].values, nsamps) #sample temp
    SP = importance_sampler(df['Salinity'].values, nsamps) #sample salt
    wind = np.random.normal(10, 2, nsamps) #sample wind
    eff = np.random.normal(0.05, 0.01, nsamps) #sample extraction efficiency
    gppm = 1.86 #ch4 in air

    #perform the conversion
    gas_sig_uatm = apply_efficiency(ch4, eff=eff) #gas sig as uatm
    gas_sig_nM = sol.sol_SP_pt(SP, pt, gas='CH4', p_dry=gas_sig_uatm*1e-6, units='mM')*1e6 #gas sig as nM
    print(np.nanmean(gas_sig_nM))
    # gas_sig_uatm = apply_efficiency(ch4, eff=eff) #gas sig as uatm
    # gas_sig_nM = sol.sol_SP_pt(SP, pt, gas='CH4', p_dry=gas_sig_uatm, units='mM') #gas sig as nM

    #compute flux
    flux = airsea.fsa_pC(gas_sig_uatm, gppm, wind, SP, pt, gas='CH4') #umol
    flux_per_day = flux*60*60*24.
    flux_per_year = flux_per_day*365.

    plt.scatter(gas_sig_nM, flux_per_year)
    plt.show()

    #compute relevant statistics
    fmean = np.mean(flux_per_year)
    fstd = np.std(flux_per_year)
    fmed = np.median(flux_per_year)
    fq1 = np.percentile(flux_per_year, 25)
    fq3 = np.percentile(flux_per_year, 75)

    print(fmean, fstd, fmed, fq1, fq3)

def compute_flux_bounds(target_mean, df, nsamps):
    ''' computes the bounds on flux for a given dataset and number of samples '''
    pt = importance_sampler(df['Temperature'].values, nsamps) #sample temp
    SP = importance_sampler(df['Salinity'].values, nsamps) #sample salt
    wind = np.random.normal(10, 2, nsamps) #sample wind
    eff = np.random.normal(0.05, 0.01, nsamps) #sample extraction efficiency
    gppm = 1.86 #ch4 in air

    #perform the conversion
    gas_sig_uatm = apply_efficiency(target_mean, eff=0.0509) #gas sig as uatm
    gas_sig_nM = sol.sol_SP_pt(SP, pt, gas='CH4', p_dry=gas_sig_uatm*1e-6, units='mM')*1e6 #gas sig as nM
    print(np.nanmean(gas_sig_nM))

    flux = airsea.fsa_pC(gas_sig_uatm, gppm, 10, 32.96, 12.7, gas='CH4') *1e6
    flux_per_day = flux*60*60*24.
    flux_per_year = flux_per_day*365.

    plt.scatter(gas_sig_uatm, flux_per_year)
    plt.show()

    #compute flux
    fpymean = np.nanmean(flux_per_year)
    fpystd = np.nanstd(flux_per_year)
    fpymed = np.nanmedian(flux_per_year)
    fpyq1 = np.percentile(flux_per_year,25)
    fpyq3 = np.percentile(flux_per_year,75)

    print('Flux per year: ', fpymean, '+/-', fpystd)
    print('Flux per year: ', fpymed, ' with ', [fpyq1, fpyq3])


if __name__ == '__main__':
    # Data to access
    base_path = './missions/falkor/'
    miss = ['Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv']

    # Visualization params
    matplotlib.rcParams['figure.figsize'] = (15,15)
    matplotlib.rcParams['font.size'] = 24
    matplotlib.rcParams['figure.titlesize'] = 28
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['axes.labelsize'] = 28
    matplotlib.rcParams['legend.fontsize'] = 24
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['savefig.directory'] = base_path

    # Create mission operator
    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_arctic_0.csv', base_path+'trimmed_arctic_2.csv'], header=0, simplify_mission=False)

    # Isolate relevant depths and different sites
    yachats = jy.mission[0]
    yachats_5m = yachats[(yachats['Depth'] < 5.0) & (yachats['Depth'] > 0.5)]
    yachats_1m = yachats[(yachats['Depth'] < 1.5) & (yachats['Depth'] > 0.5)]

    stonewall = jy.mission[1]
    stonewall_5m = stonewall[(stonewall['Depth'] < 5.0) & (stonewall['Depth'] > 0.5)]
    stonewall_1m = stonewall[(stonewall['Depth'] < 1.5) & (stonewall['Depth'] > 0.5)]

    all_samps = stonewall.append(yachats)

    # Compute the t-test for mean
    significance = 0.01
    target = yachats_1m['CH4_nM'].dropna()
    samp_mean = np.nanmean(target)
    samp_std = np.nanstd(target)
    num_samps = len(target)
    t_stat = (samp_mean - 2.7)/(samp_std/num_samps)
    t, p = sp.stats.ttest_1samp(target, 2.7)
    print('Sample mean: ', samp_mean)
    print('Significance value: ', p/2, ' Reject Null Hypothesis for Greater-Than Test: ', p/2 < 0.01)

    # Understand bounds for ppm to nM conversion
    # nsamps = 5000
    # r, means, stdevs, med, q1, q3 = run_ppm_nM_simulation(all_samps, nsamps)

    # plt.plot(r, means)
    # plt.fill_between(r, means - 2*stdevs, means + 2*stdevs, alpha=0.5)
    # plt.xlabel('CH4 Measurement, ppm')
    # plt.ylabel('Corrected CH4 Measurements, nM')
    # plt.show()

    # plt.plot(r, med)
    # plt.fill_between(r, q1, q3, alpha=0.5)
    # plt.xlabel('CH4 Measurement, ppm')
    # plt.ylabel('Corrected CH4 Measurements, nM')
    # plt.grid()
    # plt.show()

    # Understand bounds for flux
    nsamps = 50000
    compute_flux_bounds(np.nanmean(yachats_1m['CH4_ppm'].dropna()), yachats_1m, nsamps)
    # ch4 = importance_sampler(yachats_1m['CH4_ppm'].values, nsamps) + np.random.normal(0, 0.0035, nsamps) #sample ch4
    # pt = importance_sampler(yachats_1m['Temperature'].values, nsamps) #sample temp
    # SP = importance_sampler(yachats_1m['Salinity'].values, nsamps) #sample salt
    # wind = np.random.normal(10, 2, nsamps) #sample wind
    # eff = np.random.normal(0.05, 0.01, nsamps) #sample extraction efficiency

    # # run samples through simulation
    # flux_samps = []
    # gppm = 1.86 #ch4 in air

    # K0 = sol.sol_SP_pt(SP, pt, gas='CH4', units='mM')
    # gas_sig = apply_efficiency(ch4, eff=eff) #gas sig as uatm
    # plt.scatter(ch4, gas_sig, c=eff)
    # plt.show()
    # plt.hist(gas_sig, bins=100)
    # plt.xlabel('Adjusted Gas Signal for 2.7 level, uatm')
    # plt.ylabel('Count')
    # plt.show()

    # print(np.nanmean(gas_sig), np.nanstd(gas_sig))
    # print(np.nanmedian(gas_sig), sp.stats.iqr(gas_sig))

    # flux = airsea.fsa_pC(gas_sig, gppm, wind, SP, pt, gas='CH4') * 1e6
    # flux_per_day = flux*60*60*24.
    # flux_per_year = flux_per_day * 365.
    
    # plt.scatter(ch4, flux_per_year, c=eff)
    # plt.plot(ch4, np.zeros(ch4.size), c='r')
    # plt.xlabel('Methane, Measured ppm')
    # plt.ylabel('Flux, umol/m2/y')
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Efficiency')
    # plt.show()

    # idx03 = np.where(eff==0.03)[0]
    # idx05 = np.where(eff==0.05)[0]
    # idx07 = np.where(eff==0.07)[0]

    # plt.hist(flux_per_year[idx03], bins=100)
    # plt.ylabel('Number Samples')
    # plt.xlabel('Flux Value')
    # plt.show()

    # print(np.nanmean(flux_per_year[idx03]), np.nanstd(flux_per_year[idx03]))
    # print(np.nanmedian(flux_per_year[idx03]), sp.stats.iqr(flux_per_year[idx03]))

    # temp = yachats_1m['CH4_ppm'].values
    # plt.hist(temp[~np.isnan(temp)], bins=100)
    # plt.show()

    

    


    
    

