#!/usr/env/python

'''
This is an example in which all JetYak data collected is compiled.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import numpy as np
import jetyak
import jviz
import shapefile
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as mb
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.mlab import griddata
import utm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import LogFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gasex import airsea
from gasex.diff import schmidt

def lat2str(deg):
    min = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    dir = 'N'
    if deg < 0:
        if min != 0.0:
            deg += 1.0
            min -= 60.0
        dir = 'S'
    return ("%d$\degree$ %g' N") % (np.abs(deg),np.abs(min))

def lon2str(deg):
    min = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    dir = 'E'
    if deg < 0:
        if min != 0.0:
            deg += 1.0
            min -= 60.0
        dir = 'W'
    return ("%d$\degree$ %g' W") % (np.abs(deg),np.abs(min))

def get_distance(coord1, coord2):
    '''Method to get the distance in meters between two points'''
    try:
        e1, n1, zn1, zl1 = utm.from_latlon(coord1[0], coord1[1])
        e2, n2, zn2, zl2 = utm.from_latlon(coord2[0], coord2[1])
        dist = (e1-e2)**2 + (n1-n2)**2
        return np.sqrt(dist)
    except:
        return None

def add_sizebar(ax, size):
    asb = AnchoredSizeBar(ax.transData,
                          size,
                          str(size),
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)

def print_stats(ds):
    mini = np.nanmin(ds)
    maxi = np.nanmax(ds)
    med = np.nanmedian(ds)
    q75, q25 = np.nanpercentile(ds, [75, 25])
    iqr = q75 - q25
    mean = np.nanmean(ds)
    std = np.nanstd(ds)

    print 'Minimum', mini
    print 'Maximum', maxi
    print 'Median: ', med
    print 'IQR: ', q25, q75, iqr
    print 'Mean: ', mean
    print 'Stdev: ', std

    return mini, maxi, q25, med, q75, iqr, mean, std

if __name__ == '__main__':

    ####################################################
    ###### Mission Data and Params #####################
    ####################################################

    ###### Falkor Raw Data and Cleaning Params #######
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
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/falkor/'
    miss = ['Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv']

    # Create mission operator
    # jy = jetyak.JetYak()
    # jy.load_mission([base_path+m for m in miss], header=[0,1], meth_eff=0.0509)
    # jy.save_mission(base_path, mission_name='trimmed_arctic')

    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_arctic_0.csv', base_path+'trimmed_arctic_2.csv'], header=0, simplify_mission=False)

    # Inspect raw ppm values
    jy_mission_a = jy.mission[0][(jy.mission[0]['Depth'] <= 1.5) & (jy.mission[0]['Depth'] > 0.1)]
    jy_mission_b = jy.mission[1][(jy.mission[1]['Depth'] <= 1.5) & (jy.mission[1]['Depth'] > 0.1)]

    a_ch4_ppm = jy_mission_a['CH4_ppm'].values
    a_sal = list(jy_mission_a['Salinity'].values)
    a_temp = list(jy_mission_a['Temperature'].values)


    b_ch4_ppm = jy_mission_b['CH4_ppm'].values
    b_sal = list(jy_mission_b['Salinity'].values)
    b_temp = list(jy_mission_b['Temperature'].values)


    all_ch4_ppm = list(a_ch4_ppm) + list(b_ch4_ppm)
    all_sal = list(jy_mission_a['Salinity'].values) + list(jy_mission_b['Salinity'].values)
    all_temp = list(jy_mission_a['Temperature'].values) + list(jy_mission_b['Temperature'].values)

    print 'Yachats: '
    a_mini, a_maxi, a_q25, a_med, a_q75, a_iqr, a_mean, a_std = print_stats(a_ch4_ppm)
    print '----------'
    print 'Stonewall: '
    b_mini, b_maxi, b_q25, b_med, b_q75, b_iqr, b_mean, b_std = print_stats(b_ch4_ppm)
    print '----------'
    print 'All: '
    mini, maxi, q25, med, q75, iqr, mean, std = print_stats(all_ch4_ppm)


    # sns.distplot(a_ch4_ppm, label='Yachats', kde=False, bins=50,
    #              hist_kws=dict(log=True, range=(np.nanmin(a_ch4_ppm), np.nanmax(a_ch4_ppm))))
    # sns.distplot(b_ch4_ppm, label='Yachats', kde=False, bins=50,
    #              hist_kws=dict(log=True, range=(np.nanmin(b_ch4_ppm), np.nanmax(b_ch4_ppm))))
    # plt.show()

    # sns.distplot(all_ch4_ppm, label='Yachats', kde=False, bins=50,
    #              hist_kws=dict(log=True, range=(np.nanmin(all_ch4_ppm), np.nanmax(all_ch4_ppm))))
    # plt.show()

    # calculate extraction efficiency for set ppm and equilibrium value
    gppm = 1.86 #atmo levels
    peq = 495 #gga pressure
    goal = 2.7 #equilibrium nM value
    x = a_q75 #intermediate guess
    y = a_q75 #ppm we would like to have be at value x
    sal = np.nanmedian(a_sal) #salinity at the measurement y
    temp = np.nanmedian(a_temp) #temeprature at measurement y
    tol = 1e-18 #tolerance for guess

    err = 1000 #error
    f = jetyak.determine_methane

    while err > tol:
        print err
        slope = (f(sal=sal, temp=temp, fCH4=x+0.01, units='mM') - f(sal=sal, temp=temp, fCH4=x-0.01, units='mM'))/0.02
        x = x - (jetyak.determine_methane(sal=sal, temp=temp, fCH4=x, units='mM')-2.7*1e-6)/slope
        err = np.abs(2.7*1e-6 - f(sal=sal, temp=temp, fCH4=x, units='mM'))

    eff = (y - gppm)/(1000*x/peq*1e6 - gppm)
    print eff
    print f(jetyak.apply_efficiency(y, eff=eff)*1e-6, sal=sal, temp=temp)*1e6

    print '-----'

    err = 1000
    eff = 0.01
    it = 0.00001
    tol = 0.001
    sal = a_sal
    temp = a_temp
    x = a_ch4_ppm
    while err > tol:
        uatm = jetyak.apply_efficiency(np.array(x), eff=eff)*1e-6
        meth = f(sal=sal, temp=temp, fCH4=uatm)*1e6

        medi = np.nanpercentile(meth, 75)
        # print medi
        eff = eff + it

        err = np.abs(2.7 - medi)

    print eff







