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
from gasex import sol
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

if __name__ == '__main__':

    # print airsea.fsa_pC(pC_w=4e-6,pC_a=2.7e-6,u10=10,SP=32.96,T=12.7,gas='CH4',param="W14")*60*60*24*365
    # Sc = schmidt(32.96, 12.7,gas='CH4')
    winds = [0, 2, 4, 6, 8, 10]
    meths = [2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.15]
    
    for i, w in enumerate(winds):
        wvec = []
        for j, m in enumerate(meths):
            K0 = sol.sol_SP_pt(32.96, 12.7, gas='CH4', units='mM')
            gas_sig = m*1e-6  / K0 * 1e6
            flux = airsea.fsa_pC(gas_sig, 1.86, w, 32.96, 12.7, gas='CH4')
            flux_per_day = flux*60*60*24.
            flux_per_year = flux_per_day * 365.
            wvec.append(flux_per_year*1e6)
        plt.semilogy(meths, wvec, label=w)
    plt.xlabel('Methane, Concentration nM')
    plt.ylabel('Flux, umol/m2/y')
    plt.legend()
    plt.show()

    print flux*1e6
    print flux_per_day*1e6
    print flux_per_year*1e6
    print '----'

    # k = airsea.kgas(20,Sc,'W14')
    # print k * (8.15e-6 - 2.7e-6) * 60 * 60 * 24 * 365


    # take the measurement, convert to mM, divide by K0 this gives atm, convert to uatm, plug in to fsa_pC
    # 6 mM/day sanity check

    # print k * sol.sol_SP_pt(32.96, 12.7, gas='CH4', units='mM')/1e6*(2.10 - 1.86) * 60 * 60 * 24 * 365
    # flux = airsea.fsa_pC(2.10, 1.86, 10, 32.96, 12.7, gas='CH4')
    # print flux * 60 * 60 * 24 * 365 #* 16.04 / 1000.* 2870000


    # flux = airsea.fsa_pC(4.0, 1.86, 10, 32.96, 12.7, gas='CH4')
    # print flux * 60 * 60 * 24 * 365 #* 16.04 / 1000.* 2870000

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
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/falkor/'
    miss = ['Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv']
    matplotlib.rcParams['figure.figsize'] = (15,15)
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['figure.titlesize'] = 24
    matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['legend.fontsize'] = 16
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['savefig.directory'] = base_path


    # Create mission operator
    # jy = jetyak.JetYak()
    # jy.load_mission([base_path+m for m in miss], header=[0,1], meth_eff=0.0509)
    # jy.save_mission(base_path, mission_name='trimmed_arctic')

    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_arctic_0.csv', base_path+'trimmed_arctic_2.csv'], header=0, simplify_mission=False)

    # sns.distplot(jy.mission[0]['CH4_nM'], label='Yachats', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(jy.mission[0]['CH4_nM']), np.nanmax(jy.mission[0]['CH4_nM']))))
    # sns.distplot(jy.mission[1]['CH4_nM'], label='Stonewall Bank', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(jy.mission[1]['CH4_nM']), np.nanmax(jy.mission[1]['CH4_nM']))))
    # plt.axvline(2.7, 0, 10, c='r')
    # plt.legend()
    # plt.show()
    # plt.close()
    temp = jy.mission[1]
    dtemp = temp[temp['Depth'] < 1.5]
    dtemp2 = jy.mission[0][jy.mission[0]['Depth'] < 1.5]
    print np.nanmean(dtemp['CH4_nM'].values)
    print np.nanmean(dtemp2['CH4_uatm'].values)
    print 'here', np.nanmean(np.append(dtemp['CH4_nM'].values, dtemp2['CH4_nM'].values))
    print 'here', np.nanmax(np.append(dtemp['CH4_nM'].values, dtemp2['CH4_nM'].values))
    print np.nanmin(dtemp['CH4_ppm'].values)
    # asasd

    all_dist = []
    num_samples = 0
    for m in jy.mission:
        num_samples += len(m.index)
        last_point = (m['Latitude'].values[0], m['Longitude'].values[0])
        total_dist = 0
        for sample in range(len(m.index)):
            try:
                total_dist += np.fabs(get_distance(last_point, (m['Latitude'][sample], m['Longitude'][sample])))
            except:
                pass
            last_point = (m['Latitude'][sample], m['Longitude'][sample])
        all_dist.append(total_dist)
    print all_dist
    print np.sum(all_dist)
    print np.mean(all_dist)
    print np.std(all_dist)
    print num_samples

    # m = jy.mission[0]
    # fig = plt.figure(figsize=(5,5))
    # tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.5)]
    # plt.scatter(tmp['O2Concentration'], tmp['CH4_nM'], alpha=0.2)
    # plt.xlabel('O$_2$ ($\mu$M)')
    # plt.ylabel('CH$_4$ (nM)')
    # fig.subplots_adjust(bottom=0.15, top=0.95, left=0.2, right=0.9)
    # plt.show()


    target='CH4_nM'
    target_constant = 1

    for m in [jy.mission[1]]: #Stonewall Bank
        # select which part of the watercolumn to analyze
        tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)] #surface
        ctmp = m[m['Depth'] > 0.5] #water column
        ctmp.to_csv(base_path+'stonewall_depth_controlled.csv')

        plt.plot(ctmp['Temperature'].values)
        plt.show()

        # plotting helpers
        vmin = np.nanmin(tmp[target].values)
        vmax = np.nanmax(tmp[target].values)
        x_min = np.nanmin(tmp['Latitude']) - 0.001
        x_max = np.nanmax(tmp['Latitude']) + 0.001
        y_min = np.nanmin(tmp['Longitude']) - 0.001
        y_max = np.nanmax(tmp['Longitude']) + 0.001

        rew = get_distance((x_min, y_min), (x_min, y_max))
        req = get_distance((x_min, y_min), (x_max, y_min))
        print rew*req

        #generate joint plots
        j = sns.jointplot(ctmp['CH4_nM'], ctmp['Depth'], label='Measurements', height=15, ratio=3, s=10, marginal_kws=dict(bins=50, kde=True))
        target = 'CH4_nM'
        avgs = []
        stds = []
        ds = []
        depths = [0.5+i*0.25 for i in range(0,100)]
        last_depth = depths[0]
        for d in depths:
            temp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
            avgs.append(np.mean(temp[target]))
            stds.append(np.std(temp[target]))
            ds.append(np.mean(temp['Depth']))
            last_depth = d
        j.x = avgs
        j.y = ds
        # j.xerr = stds
        j.plot_joint(plt.errorbar, color='r', label='Average at Depth', markersize=5, marker='o', lw=0)#, xerr=stds)
        j.fig.set_figwidth(15)
        j.fig.set_figheight(6)
        j.fig.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)
        plt.axvline(2.7, 0, 10, c='k', label='Equilibrium Concentration', zorder=10)
        j.set_axis_labels('Methane, nM', 'Depth, m')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()

        #generate histograms
        j = sns.distplot(tmp['CH4_nM'], label='Surface Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(tmp['CH4_nM']), np.nanmax(tmp['CH4_nM']))))
        sns.distplot(ctmp['CH4_nM'], label='All Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(ctmp['CH4_nM']), np.nanmax(ctmp['CH4_nM']))))
        plt.axvline(2.7, 0, 10, c='r')
        plt.legend()
        plt.show()
        plt.close()

        #generate slice and map
        # fig = plt.figure(figsize=(15, 25))
        # spec = gridspec.GridSpec(nrows=10, ncols=10, bottom=0.08, top=0.89, left=0.1, right=0.90, hspace=0.15)
        # ax = fig.add_subplot(spec[-2:, 2:8])
        # scat = ax.scatter(ctmp.index-ctmp.index[0], ctmp['Depth'], c=ctmp['CH4_nM'], cmap='viridis', s=5.0, norm=matplotlib.colors.LogNorm())
        # ax.set_xlabel('Sample Number')
        # ax.set_ylabel('Depth, m')
        # ax.set_yticks(np.arange(0, 12, 2))
        # ax.set_ylim(0,10.)
        # ax.invert_yaxis()

        # ax2 = fig.add_subplot(spec[0:-2, :])
        # m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        # tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        # m_tmp = ctmp.sort_values(by='CH4_nM', ascending=True)
        # plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        # casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        # base = Basemap(llcrnrlon=y_min,
        #                llcrnrlat=x_min,
        #                urcrnrlon=y_max,
        #                urcrnrlat=x_max,
        #                resolution='l',
        #                projection='cyl',
        #                suppress_ticks=False,
        #                ax=ax2)

        # proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        # path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        # scat = base.scatter(path_lon,
        #                     path_lat,
        #                     zorder=5,
        #                     s=30.0,
        #                     alpha=1.0,
        #                     c=m_tmp[target].values,
        #                     cmap='viridis',
        #                     lw=0,
        #                     norm=matplotlib.colors.LogNorm())
        # base.scatter(-1*plume_lon, plume_lat, s=500, c='m', lw=0, alpha=0.5, zorder=3, label='Bubble plumes')
        # base.scatter([x[1] for x in casts], [x[0] for x in casts], s=700, c='k', marker='*', zorder=6, label='CTD025')
        
        # ax = plt.gca()
        # def xformat(x, pos=None): return lon2str(x)
        # def yformat(x, pos=None): return lat2str(x)
        # ax.xaxis.tick_top()
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        # length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        # base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        # ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        # plt.setp(ax.get_xticklabels(), rotation=30, ha="left", rotation_mode="anchor")
        # plt.legend()

        # cax = fig.add_axes([0.75, 0.08, 0.025, 0.80])
        # formatter = LogFormatter(10, labelOnlyBase=False)
        # cbar = plt.colorbar(scat, ticks=[3, 11, 19, 27, 35], format=formatter, cax=cax)
        # cbar.ax.set_yticklabels([3, 11, 19, 27, 35], fontsize=14)
        # cbar.set_label('Methane, nM')

        # plt.show()
        # plt.close()

        #generate slice and map
        fig, ax = plt.subplots(figsize=(15, 25))

        m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        m_tmp = ctmp.sort_values(by='CH4_nM', ascending=True)
        plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        base = Basemap(llcrnrlon=y_min,
                       llcrnrlat=x_min,
                       urcrnrlon=y_max,
                       urcrnrlat=x_max,
                       resolution='l',
                       projection='cyl',
                       suppress_ticks=False,
                       ax=ax)

        proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        scat = base.scatter(path_lon,
                            path_lat,
                            zorder=5,
                            s=30.0,
                            alpha=1.0,
                            c=m_tmp[target].values,
                            cmap='viridis',
                            lw=0,
                            norm=matplotlib.colors.LogNorm())
        base.scatter(-1*plume_lon, plume_lat, s=500, c='m', lw=0, alpha=0.5, zorder=3, label='Bubble plumes')
        base.scatter([x[1] for x in casts], [x[0] for x in casts], s=700, c='k', marker='*', zorder=6, label='CTD025')
        
        def xformat(x, pos=None): return lon2str(x)
        def yformat(x, pos=None): return lat2str(x)
        ax.xaxis.tick_top()
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="left", rotation_mode="anchor")
        plt.legend()

        divider = make_axes_locatable(ax)
        axlon = divider.append_axes("bottom", 1.5, pad=0.1, sharex=ax)
        axlat = divider.append_axes("right", 1.5, pad=0.1, sharey=ax)
        axlon.xaxis.set_tick_params(labelbottom=False)
        axlat.yaxis.set_tick_params(labelleft=False)


        m_tmp = m_tmp.sort_values(by='CH4_nM', ascending=True)
        axlon.scatter(m_tmp['Longitude'], m_tmp['Depth'], c=m_tmp['CH4_nM'], cmap='viridis', s=5.0, norm=matplotlib.colors.LogNorm())
        axlon.set_ylabel('Depth (m)')
        axlon.set_yticks(np.arange(0, 12, 2))
        axlon.set_ylim(0,10.)
        axlon.invert_yaxis()

        axlat.scatter(m_tmp['Depth'], m_tmp['Latitude'], c=m_tmp['CH4_nM'], cmap='viridis', s=5.0, norm=matplotlib.colors.LogNorm())
        axlat.set_xlabel('Depth (m)')
        axlat.set_xticks(np.arange(0, 12, 2))
        axlat.set_xlim(0,10.)

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)
        
        cax = fig.add_axes([0.85, 0.08, 0.025, 0.80])
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar = plt.colorbar(scat, ticks=[3, 11, 19, 27, 35], format=formatter, cax=cax)
        cbar.ax.set_yticklabels([3, 11, 19, 27, 35], fontsize=14)
        cbar.set_label('Methane, nM')

        plt.show()
        plt.close()

        #generate thresholded surface plot
        # tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)]
        # m_tmp = tmp[tmp['CH4_nM'] >= 2.7]
        # tmp = tmp[tmp['CH4_nM'] < 2.7]
        # plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        # casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        # base = Basemap(llcrnrlon=y_min,
        #                llcrnrlat=x_min,
        #                urcrnrlon=y_max,
        #                urcrnrlat=x_max,
        #                resolution='l',
        #                projection='cyl',
        #                suppress_ticks=False)

        # proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        # path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        # scat = base.scatter(path_lon,
        #                     path_lat,
        #                     zorder=5,
        #                     s=30.0,
        #                     alpha=1.0,
        #                     c=m_tmp[target].values,
        #                     cmap='coolwarm',
        #                     lw=0)
        # base.scatter(proj_lon,
        #              proj_lat,
        #              zorder=3,
        #              s=30.0,
        #              alpha=1.0,
        #              c='k')
        # base.scatter(-1*plume_lon, plume_lat, s=200, c='m', lw=0, zorder=6, label='Bubble plumes')
        # base.scatter([x[1] for x in casts], [x[0] for x in casts], s=300, c='k', marker='*', zorder=6, label='CTD025')
        
        # cbar = plt.colorbar(scat)
        # cbar.set_label('Methane, nM')
        
        # ax = plt.gca()
        # def xformat(x, pos=None): return lon2str(x)
        # def yformat(x, pos=None): return lat2str(x)
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        # length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        # base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        # ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # plt.gcf().tight_layout(rect=(0,0.11,1,1))

        # plt.legend()
        # plt.show()
        # plt.close()


    for m in [jy.mission[0]]:
        # select which part of the watercolumn to analyze
        tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)] #surface
        ctmp = m[m['Depth'] > 0.5] #water column
        ctmp.to_csv(base_path+'yachats_depth_controlled.csv')

        # print interesting stats
        print np.mean(m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)]['Salinity'].values)
        print np.mean(m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)]['Temperature'].values)
        print '-------'

        # plotting helpers
        vmin = np.nanmin(tmp[target].values)
        vmax = np.nanmax(tmp[target].values)
        x_min = np.nanmin(tmp['Latitude']) - 0.001
        x_max = np.nanmax(tmp['Latitude']) + 0.001
        y_min = np.nanmin(tmp['Longitude']) - 0.001
        y_max = np.nanmax(tmp['Longitude']) + 0.001

        #generate joint plots
        j = sns.jointplot(ctmp['CH4_nM'], ctmp['Depth'], label='Measurements', height=15, ratio=3, s=10, marginal_kws=dict(bins=50, kde=True))
        target = 'CH4_nM'
        avgs = []
        stds = []
        ds = []
        depths = [0.5+i*0.25 for i in range(0,100)]
        last_depth = depths[0]
        for d in depths:
            temp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
            avgs.append(np.mean(temp[target]))
            stds.append(np.std(temp[target]))
            ds.append(np.mean(temp['Depth']))
            last_depth = d
        j.x = avgs
        j.y = ds
        # j.xerr = stds
        j.plot_joint(plt.errorbar, color='r', label='Average at Depth', markersize=5, marker='o', lw=0)#, xerr=stds)
        j.fig.set_figwidth(15)
        j.fig.set_figheight(6)
        j.fig.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)
        plt.axvline(2.7, 0, 10, c='k', label='Equilibrium Concentration', zorder=10)
        j.set_axis_labels('Methane, nM', 'Depth, m')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()

        #generate kde plots
        j = sns.distplot(tmp['CH4_nM'], label='Surface Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(tmp['CH4_nM']), np.nanmax(tmp['CH4_nM']))))
        sns.distplot(ctmp['CH4_nM'], label='All Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(ctmp['CH4_nM']), np.nanmax(ctmp['CH4_nM']))))
        plt.axvline(2.7, 0, 10, c='r')
        plt.legend()
        plt.show()
        plt.close()

        #generate slice and map
        # fig = plt.figure(figsize=(15, 25))
        # spec = gridspec.GridSpec(nrows=10, ncols=10, bottom=0.08, top=0.89, left=0.1, right=0.90, hspace=0.15)
        # ax = fig.add_subplot(spec[-2:, 2:8])
        # scat = ax.scatter(ctmp.index-ctmp.index[0], ctmp['Depth'], c=ctmp['CH4_nM'], cmap='viridis', s=5.0, norm=matplotlib.colors.LogNorm())
        # ax.set_xlabel('Sample Number')
        # ax.set_ylabel('Depth, m')
        # ax.set_yticks(np.arange(0, 12, 2))
        # ax.set_ylim(0,10.)
        # ax.invert_yaxis()

        # ax2 = fig.add_subplot(spec[0:-2, :])
        # m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        # tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        # m_tmp = ctmp.sort_values(by='CH4_nM', ascending=True)
        # plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        # casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        # base = Basemap(llcrnrlon=y_min,
        #                llcrnrlat=x_min,
        #                urcrnrlon=y_max,
        #                urcrnrlat=x_max,
        #                resolution='l',
        #                projection='cyl',
        #                suppress_ticks=False,
        #                ax=ax2)

        # proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        # path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        # scat = base.scatter(path_lon,
        #                     path_lat,
        #                     zorder=5,
        #                     s=30.0,
        #                     alpha=1.0,
        #                     c=m_tmp[target].values,
        #                     cmap='viridis',
        #                     lw=0,
        #                     norm=matplotlib.colors.LogNorm())
        # base.scatter(-1*plume_lon, plume_lat, s=500, c='m', lw=0, alpha=0.5, zorder=3, label='Bubble plumes')
        # # base.scatter([x[1] for x in casts], [x[0] for x in casts], s=700, c='k', marker='*', zorder=6, label='CTD025')
        
        # ax = plt.gca()
        # def xformat(x, pos=None): return lon2str(x)
        # def yformat(x, pos=None): return lat2str(x)
        # ax.xaxis.tick_top()
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        # length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        # base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        # ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        # plt.setp(ax.get_xticklabels(), rotation=30, ha="left", rotation_mode="anchor")
        # plt.legend()

        # cax = fig.add_axes([0.75, 0.08, 0.025, 0.80])
        # formatter = LogFormatter(10, labelOnlyBase=False)
        # cbar = plt.colorbar(scat, ticks=[2, 3, 4, 5, 6], format=formatter, cax=cax)
        # cbar.ax.set_yticklabels([2, 3, 4, 5, 6], fontsize=14)
        # cbar.set_label('Methane, nM')

        # plt.show()
        # plt.close()

        # generate slices and plots
        fig, ax = plt.subplots(figsize=(15, 25))

        m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        m_tmp = ctmp.sort_values(by='CH4_nM', ascending=True)
        plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        base = Basemap(llcrnrlon=y_min,
                       llcrnrlat=x_min,
                       urcrnrlon=y_max,
                       urcrnrlat=x_max,
                       resolution='l',
                       projection='cyl',
                       suppress_ticks=False,
                       ax=ax)

        proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        scat = base.scatter(path_lon,
                            path_lat,
                            zorder=5,
                            s=30.0,
                            alpha=1.0,
                            c=m_tmp[target].values,
                            cmap='viridis',
                            lw=0,
                            norm=matplotlib.colors.LogNorm())
        base.scatter(-1*plume_lon, plume_lat, s=500, c='m', lw=0, alpha=0.5, zorder=3, label='Bubble plumes')
        base.scatter([x[1] for x in casts], [x[0] for x in casts], s=700, c='k', marker='*', zorder=6, label='CTD025')
        
        def xformat(x, pos=None): return lon2str(x)
        def yformat(x, pos=None): return lat2str(x)
        ax.xaxis.tick_top()
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="left", rotation_mode="anchor")
        plt.legend()

        divider = make_axes_locatable(ax)
        axlon = divider.append_axes("bottom", 1.5, pad=0.1, sharex=ax)
        axlat = divider.append_axes("right", 1.5, pad=0.1, sharey=ax)
        axlon.xaxis.set_tick_params(labelbottom=False)
        axlat.yaxis.set_tick_params(labelleft=False)


        m_tmp = m_tmp.sort_values(by='CH4_nM', ascending=True)
        axlon.scatter(m_tmp['Longitude'], m_tmp['Depth'], c=m_tmp['CH4_nM'], cmap='viridis', s=5.0, norm=matplotlib.colors.LogNorm())
        axlon.set_ylabel('Depth (m)')
        axlon.set_yticks(np.arange(0, 12, 2))
        axlon.set_ylim(0,10.)
        axlon.invert_yaxis()

        axlat.scatter(m_tmp['Depth'], m_tmp['Latitude'], c=m_tmp['CH4_nM'], cmap='viridis', s=5.0, norm=matplotlib.colors.LogNorm())
        axlat.set_xlabel('Depth (m)')
        axlat.set_xticks(np.arange(0, 12, 2))
        axlat.set_xlim(0,10.)

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)
        
        cax = fig.add_axes([0.85, 0.08, 0.025, 0.80])
        formatter = LogFormatter(10, labelOnlyBase=False)
        cbar = plt.colorbar(scat, ticks=[2, 3, 4, 5, 6, 7], format=formatter, cax=cax)
        cbar.ax.set_yticklabels([2, 3, 4, 5, 6, 7], fontsize=14)
        cbar.set_label('Methane, nM')

        plt.show()
        plt.close()

        # #generate slices
        # scat = plt.scatter(ctmp.index-ctmp.index[0], ctmp['Depth'], c=ctmp['CH4_nM'], cmap='coolwarm', s=10.0)
        # plt.gca().invert_yaxis()
        # cbar = plt.colorbar(scat)
        # cbar.set_label('Methane, nM')
        # plt.xlabel('Sample Number')
        # plt.ylabel('Depth, m')
        # plt.show()
        # plt.close()

        # #generate thresholded flattened plot
        # m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        # tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        # plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        # casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        # base = Basemap(llcrnrlon=y_min,
        #                llcrnrlat=x_min,
        #                urcrnrlon=y_max,
        #                urcrnrlat=x_max,
        #                resolution='l',
        #                projection='cyl',
        #                suppress_ticks=False)
        # proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        # path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        # scat = base.scatter(path_lon,
        #                     path_lat,
        #                     zorder=5,
        #                     s=30.0,
        #                     alpha=1.0,
        #                     c=m_tmp[target].values,
        #                     cmap='coolwarm',
        #                     lw=0)
        # base.scatter(proj_lon,
        #              proj_lat,
        #              zorder=3,
        #              s=30.0,
        #              alpha=1.0,
        #              c='k')
        # base.scatter(-1*plume_lon, plume_lat, s=200, c='m', lw=0, zorder=6, label='Bubble plumes')
        # # base.scatter([x[1] for x in casts], [x[0] for x in casts], s=300, c='k', marker='*', zorder=6, label='CTD025')
        
        # cbar = plt.colorbar(scat)
        # cbar.set_label('Methane, nM')
        
        # ax = plt.gca()
        # def xformat(x, pos=None): return lon2str(x)
        # def yformat(x, pos=None): return lat2str(x)
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        # length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        # base.plot([y_max-0.00226, y_max-0.001], [x_min+0.001, x_min+0.001], marker='|', c='k')
        # ax.annotate(str(int(length))+'m', (np.mean([y_max-0.00226, y_max-0.001])-0.0002, x_min+0.00105), fontsize=14)
        # ax.set_xlabel('')
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # plt.gcf().tight_layout(rect=(0,0.11,1,1))
        # plt.legend()
        # plt.show()
        # plt.close()

        # #generate thresholded surface plot
        # tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)]
        # m_tmp = tmp[tmp['CH4_nM'] >= 2.7]
        # tmp = tmp[tmp['CH4_nM'] < 2.7]
        # plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        # casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        # base = Basemap(llcrnrlon=y_min,
        #                llcrnrlat=x_min,
        #                urcrnrlon=y_max,
        #                urcrnrlat=x_max,
        #                resolution='l',
        #                projection='cyl',
        #                suppress_ticks=False)

        # proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        # path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        # scat = base.scatter(path_lon,
        #                     path_lat,
        #                     zorder=5,
        #                     s=30.0,
        #                     alpha=1.0,
        #                     c=m_tmp[target].values,
        #                     cmap='coolwarm',
        #                     lw=0)
        # base.scatter(proj_lon,
        #              proj_lat,
        #              zorder=3,
        #              s=0.3,
        #              alpha=0.1,
        #              c='k')
        # base.scatter(-1*plume_lon, plume_lat, s=200, c='m', lw=0, zorder=6, label='Bubble plumes')
        # # base.scatter([x[1] for x in casts], [x[0] for x in casts], s=300, c='k', marker='*', zorder=6, label='CTD025')
        
        # cbar = plt.colorbar(scat)
        # cbar.set_label('Methane, nM')
        
        # ax = plt.gca()
        # def xformat(x, pos=None): return lon2str(x)
        # def yformat(x, pos=None): return lat2str(x)
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        # length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        # base.plot([y_max-0.00226, y_max-0.001], [x_min+0.001, x_min+0.001], marker='|', c='k')
        # ax.annotate(str(int(length))+'m', (np.mean([y_max-0.00226, y_max-0.001])-0.0002, x_min+0.00105), fontsize=14)
        # ax.set_xlabel('')
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # plt.gcf().tight_layout(rect=(0,0.11,1,1))
        # plt.legend()
        # plt.show()
        # plt.close()
