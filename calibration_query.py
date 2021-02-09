#!/usr/env/python

'''
How to use the JetYak GGA and DGEU models to extract data from calibration code.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

# base libraries
import numpy as np
import scipy as sp
import jetyak
import utm
from gasex import airsea
from gasex import sol
from gasex.diff import schmidt

# visualization functionality
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.mlab import griddata
import matplotlib.gridspec as gridspec
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
    hx, hy, _ = plt.hist(data, bins=100, normed=1)
    dx = hy[1] - hy[0]
    F1 = np.cumsum(hx)*dx
    #uniformly sample from 0 to 1
    n = np.random.uniform(0, 1, nsamps)
    #query the cdf to get the potential generating value
    samps = np.interp(n, F1, hy[1:])
    plt.close()
    return samps


if __name__ == '__main__':
    # simple flux model
    # winds = [0, 2, 4, 6, 8, 10]
    # meths = [2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.15]

    # SP = 32.96 #average salinity
    # pt = 12.7 #average potential temperature
    # gppm = 1.86 #ch4 in air
    # wind = 6 #average wind from gauge
    # effs = [0.03, 0.05, 0.07] #extraction efficiency
    # std_dev = 0.0035 #standard deviation from lab measurements in air

    
    # for i, eff in enumerate(effs):
    #     vec = []
    #     vec_low = []
    #     vec_high = []
    #     for j, m in enumerate(meths):
    #         K0 = sol.sol_SP_pt(SP, pt, gas='CH4', units='mM')
    #         gas_sig = apply_efficiency(m, eff=eff) #gas sig as uatm
    #         gas_sig_low = apply_efficiency(m-2*std_dev, eff=eff)
    #         gas_sig_high = apply_efficiency(m+2*std_dev, eff=eff)
    #         flux = airsea.fsa_pC(gas_sig, gppm, wind, SP, pt, gas='CH4') * 1e6
    #         flux_low = airsea.fsa_pC(gas_sig_low, gppm, wind, SP, pt, gas='CH4') * 1e6
    #         flux_high = airsea.fsa_pC(gas_sig_high, gppm, wind, SP, pt, gas='CH4') * 1e6
    #         flux_per_day = flux*60*60*24.
    #         flux_per_day_low = flux_low*60*60*24.
    #         flux_per_day_high = flux_high*60*60*24.
    #         flux_per_year = flux_per_day * 365.
    #         flux_per_year_low = flux_per_day_low*365.
    #         flux_per_year_high = flux_per_day_high*365.
    #         print flux_low, flux, flux_high
    #         print flux_per_day_low, flux_per_day, flux_per_day_high
    #         print flux_per_year_low, flux_per_year, flux_per_year_high
    #         print '----'
    #         vec.append(flux_per_year*1e6)
    #         vec_low.append(flux_per_year_low*1e6)
    #         vec_high.append(flux_per_year_high*1e6)
    #     plt.plot(meths, vec, label=eff)
    #     plt.fill_between(meths, vec_low, vec_high, alpha=0.5)
    # plt.xlabel('Methane, Concentration nM')
    # plt.ylabel('Flux, umol/m2/y')
    # plt.yscale('log', nonposy='clip')
    # plt.legend()
    # plt.show()

    # print flux
    # print flux_per_day
    # print flux_per_year
    # print '----'

    ####################################################
    ###### Mission Data and Params #####################
    ####################################################

    ###### Falkor Data #######
    # base_path = '/home/vpreston/Documents/field_work/falkor_cruise_2018/Falkor-09.2018/09.13.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_20180914005320.txt']
    # gga_dirs = [base_path + 'gga/gga_20180914005646.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180914005404.txt']
    # optode_dirs = [base_path + 'op/optode_20180914005425.txt']
    # mission_name = 'Falkor_0913.csv'
    # trim_values = None
    # bounds = None
    # offset = 2440587.6705

    # base_path = '/home/vpreston/Documents/field_work/falkor_cruise_2018/Falkor-09.2018/09.14.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180915023820.txt', base_path + 'airmar/airmar_20180915030636.txt', base_path + 'airmar/airmar_20180915050321.txt']
    # gga_dirs = [base_path + 'gga/gga_20180915023928.txt', base_path + 'gga/gga_20180915030848.txt', base_path + 'gga/gga_20180915050416.txt']
    # optode_dirs = [base_path + 'op/optode_20180915023956.txt', base_path + 'op/optode_20180915030827.txt', base_path + 'op/optode_20180915050341.txt']
    # mission_name = 'Falkor_0914.csv'
    # trim_values = None
    # bounds = None
    # offset = 2440587.498

    # base_path = '/home/vpreston/Documents/field_work/falkor_cruise_2018/Falkor-09.2018/09.16.2018/data/'
    # ctd_dirs = [base_path + 'ctd/data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180917042510.txt']
    # gga_dirs = [base_path + 'gga/gga_20180917042451.txt']
    # optode_dirs = [base_path + 'op/optode_20180917042535.txt']
    # mission_name = 'Falkor_0916.csv'
    # trim_values = None
    # bounds = None
    # offset = 2440587.503


    ####################################################
    ###### Make a "mission" JetYak #####################
    ####################################################
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset])
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # jy.attach_sensor('optode', optode_dirs)


    # # # Can now perform work with the sensors
    # jy.create_mission({'geoframe':'airmar'})
    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/falkor/', mission_name=mission_name)
    # print jy.mission[0].head(5)

    ####################################################
    ###### Make a mission "analyzing" JetYak ###########
    ####################################################
    # Data to access
    base_path = './missions/falkor/'
    miss = ['Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv']
    matplotlib.rcParams['figure.figsize'] = (15,15)
    matplotlib.rcParams['font.size'] = 24
    matplotlib.rcParams['figure.titlesize'] = 28
    matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.labelsize'] = 28
    matplotlib.rcParams['legend.fontsize'] = 24
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['savefig.directory'] = base_path


    # Create mission operator
    # jy = jetyak.JetYak()
    # jy.load_mission([base_path+m for m in miss], header=[0,1], meth_eff=0.0509)
    # jy.save_mission(base_path, mission_name='trimmed_arctic') #use arctic reference for efficiency

    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_arctic_0.csv', base_path+'trimmed_arctic_2.csv'], header=0, simplify_mission=False)

    print '------Yachats Stats-------'
    yachats = jy.mission[0]
    yachats_5m = yachats[(yachats['Depth'] < 5.0) & (yachats['Depth'] > 0.5)]
    yachats_1m = yachats[(yachats['Depth'] < 1.5) & (yachats['Depth'] > 0.5)]
    print '------Stonewall Bank Stats-------'
    stonewall = jy.mission[1]
    stonewall_5m = stonewall[(stonewall['Depth'] < 5.0) & (stonewall['Depth'] > 0.5)]
    stonewall_1m = stonewall[(stonewall['Depth'] < 1.5) & (stonewall['Depth'] > 0.5)]

    all_5m = stonewall.append(yachats)
    print np.nanmax(all_5m['CH4_ppm'].values), np.nanmin(all_5m['CH4_ppm'].values)
    print np.nanmax(all_5m['CH4_nM'].values), np.nanmin(all_5m['CH4_nM'].values)

    # create samplers for flux, based on Yachats
    nsamps = 25000
    ch4 = importance_sampler(yachats_1m['CH4_ppm'].values, nsamps) + np.random.normal(0, 0.0035, nsamps) #np.random.normal(np.nanmean(yachats_1m['CH4_ppm'].values), np.nanstd(yachats_1m['CH4_ppm'].values), nsamps) + np.random.normal(0, 0.0035, nsamps) #sample ch4
    # ch4 = 5.0 * np.ones(ch4.size) + np.random.normal(0, 0.0035, nsamps)
    pt = importance_sampler(yachats_1m['Temperature'].values, nsamps) #np.random.normal(np.nanmean(yachats_1m['Salinity'].values), np.nanstd(yachats_1m['Salinity'].values), nsamps) #sample wind
    SP = importance_sampler(yachats_1m['Salinity'].values, nsamps) # np.random.normal(np.nanmean(yachats_1m['Temperature'].values), np.nanstd(yachats_1m['Temperature'].values), nsamps) #sample salt
    wind = np.random.normal(10, 2, nsamps) #sample wind
    eff = np.random.normal(0.05, 0.01, nsamps)#np.random.choice([0.03, 0.05, 0.07], nsamps) #choose an efficiency

    # run samples through simulation
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

    # print np.nanmean(gas_sig), np.nanstd(gas_sig)
    # print np.nanmedian(gas_sig), sp.stats.iqr(gas_sig)

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

    # print np.nanmean(flux_per_year[idx03]), np.nanstd(flux_per_year[idx03])
    # print np.nanmedian(flux_per_year[idx03]), sp.stats.iqr(flux_per_year[idx03])

    # temp = yachats_1m['CH4_ppm'].values
    # plt.hist(temp[~np.isnan(temp)], bins=100)
    # plt.show()

    all_samps = stonewall.append(yachats)
    # create samplers for flux, based on Yachats
    nsamps = 50000
    pt = importance_sampler(all_samps['Temperature'].values, nsamps) #np.random.normal(np.nanmean(yachats_1m['Salinity'].values), np.nanstd(yachats_1m['Salinity'].values), nsamps) #sample wind
    SP = importance_sampler(all_samps['Salinity'].values, nsamps) # np.random.normal(np.nanmean(yachats_1m['Temperature'].values), np.nanstd(yachats_1m['Temperature'].values), nsamps) #sample salt
    wind = np.random.normal(10, 2, nsamps) #sample wind
    eff = np.random.normal(0.05, 0.01, nsamps)#np.random.choice([0.03, 0.05, 0.07], nsamps) #choose an efficiency


    mmeans = []
    mstdevs = []
    mmed = []
    mq1 = []
    mq3 = []
    r = np.linspace(np.nanmin(all_5m['CH4_ppm'].values), 8)# np.nanmax(all_5m['CH4_ppm'].values), 50)
    for m in r:
        ch4 = m * np.ones(eff.size) + np.random.normal(0, 0.0035, nsamps)
        gas_sig = apply_efficiency(ch4, eff=eff)/0.000001 #gas sig as uatm
        gas_sig = sol.sol_SP_pt(SP, pt, gas='CH4', p_dry=gas_sig, units='mM') * 0.000001
        mmeans.append(np.nanmean(gas_sig))
        mstdevs.append(np.nanstd(gas_sig))
        mmed.append(np.nanmedian(gas_sig))
        mq1.append(np.percentile(gas_sig, 25))
        mq3.append(np.percentile(gas_sig, 75))

    mmeans = np.asarray(mmeans)
    mstdevs = np.asarray(mstdevs)

    plt.plot(r, mmeans)
    plt.fill_between(r, mmeans - 2*mstdevs, mmeans + 2*mstdevs, alpha=0.5)
    plt.xlabel('CH4 Measurement, ppm')
    plt.ylabel('Corrected CH4 Measurements, nM')
    plt.show()

    plt.plot(r, mmed)
    plt.fill_between(r, mq1, mq3, alpha=0.5)
    plt.xlabel('CH4 Measurement, ppm')
    plt.ylabel('Corrected CH4 Measurements, nM')
    plt.grid()
    plt.show()

    


    
    

