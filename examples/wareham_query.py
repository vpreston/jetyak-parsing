#!/usr/env/python

'''
The main file for creating and analyzing JetYak missions.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import numpy as np
import jetyak
import jviz
import sensors
import shapefile
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import utm
import seawater.eos80 as gsw
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as mb
from scipy import stats
from gasex import sol

def get_distance(coord1, coord2):
    '''Method to get the distance in meters between two points'''
    try:
        e1, n1, zn1, zl1 = utm.from_latlon(coord1[0], coord1[1])
        e2, n2, zn2, zl2 = utm.from_latlon(coord2[0], coord2[1])
        dist = (e1-e2)**2 + (n1-n2)**2
        if np.sqrt(dist) > 3000:
            return None
        return np.sqrt(dist) * np.sign(coord1[1]-coord2[1])
    except:
        return None

def getCircle(lat, lon, radius):
    lats = []
    lons = []

    utm_lat, utm_lon, zn, zl = utm.from_latlon(lat, lon)

    for deg in range(0, 360):
        plat = radius * np.cos(deg*np.pi/180.)+utm_lat
        plon = radius * np.sin(deg*np.pi/180.)+utm_lon
        tlat, tlon =  utm.to_latlon(plat, plon, zn, zl)
        lats.append(tlat)
        lons.append(tlon)

    return lats, lons

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



if __name__ == '__main__':

    ####################################################
    #################### LOAD DATA #####################
    ####################################################

    base_path = '/home/vpreston/Documents/field_work/wareham_2017/rawdata/'
    ctd_dirs = [base_path + 'wareham_ctd_data.txt']
    gga_dirs = [base_path + 'gga_2017-07-12_f0001.txt']
    op_dirs = [base_path + 'optode_20170712144117.txt',
               base_path + 'optode_20170712155518.txt',
               base_path + 'optode_20170712194058.txt']
    airmar_dirs = [base_path + 'airmar1.txt',
                   base_path + 'airmar2.txt',
                   base_path + 'airmar3.txt']
    mission_name = 'wareham_2017.csv'
    trim_values = None
    bounds = [1.116 + 2457946, 1.282 + 2457946]
    offset =  2440587.665

    ####################################################
    ############ PLOTTING DEFAULTS  ####################
    ####################################################
    matplotlib.rcParams['figure.figsize'] = (15,15)
    matplotlib.rcParams['font.size'] = 24
    matplotlib.rcParams['figure.titlesize'] = 28
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 22
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['savefig.directory'] = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/wareham/'


    ####################################################
    ################# PROCESS DATA #####################
    ####################################################
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset])
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # jy.attach_sensor('optode', op_dirs)

    # # Can now perform work with the sensors
    # jy.create_mission({'geoframe':'airmar'})

    # # Check everything
    # m = jy.mission[0]
    # print m.head(10)

    # plt.plot(jy.ctd.get_df()['Julian_Date'], jy.ctd.get_df()['Salinity']/np.nanmax(jy.ctd.get_df()['Salinity']))
    # plt.plot(jy.gga.get_df()['Julian_Date'], jy.gga.get_df()['CH4_ppm']/np.nanmax(jy.gga.get_df()['CH4_ppm']))
    # plt.plot(jy.optode.get_df()['Julian_Date'], jy.optode.get_df()['O2Concentration']/np.nanmax(jy.optode.get_df()['O2Concentration']))
    # plt.show()

    # plt.plot(jy.airmar.get_df()['Julian_Date'], jy.airmar.get_df()['lon_mod'])
    # plt.show()

    # # Create Transects
    # transects = [m[m.index <= 2457947.145],
    #              m[(m.index <= 2457947.19) & (m.index > 2457947.145)],
    #              m[(m.index <= 2457947.20) & (m.index > 2457947.19)],
    #              m[(m.index <= 2457947.215) & (m.index > 2457947.20)],
    #              m[(m.index <= 2457947.26) & (m.index > 2457947.215)],
    #              m[(m.index > 2457947.26)]]

    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/wareham/', mission_name=mission_name)
    
    # for i, t in enumerate(transects):
    #     t.to_csv('/home/vpreston/Documents/IPP/jetyak_parsing/missions/wareham/transect_'+str(i)+'.csv')
    
    ####################################################
    ####### READ IN PRE-PROCESSED DATA #################
    ####################################################
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/wareham/'
    miss = ['wareham_2017.csv']
    titles = ['Wareham 2017']

    ''' Create mission operator '''
    jy = jetyak.JetYak()

    ''' If the file isn't simlified or conversions made, run this '''
    # jy.load_mission([base_path+m for m in miss], header=[0,1], simplify_mission=True, meth_eff=0.15, carb_eff=0.70)
    # jy.save_mission(base_path, 'trimmed_chemyak_cleaned')

    # for i in [0,1,2,3,4,5]:
    #     jy = jetyak.JetYak()
    #     jy.load_mission([base_path+'transect_'+str(i)+'.csv'], header=[0,1], simplify_mission=True, meth_eff=0.15, carb_eff=0.70)
    #     jy.save_mission(base_path, 'trimmed_transect_'+str(i))

    ''' Read in simplified targets'''
    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_chemyak_cleaned_'+str(i)+'.csv' for i in [0]], header=0, simplify_mission=False)

    transects = []
    for i in [0, 1, 2, 3, 4, 5]:
        temp_jy = jetyak.JetYak()
        temp_jy.load_mission([base_path+'trimmed_transect_'+str(i)+'_0.csv'], header=0, simplify_mission=False)
        transects.append(temp_jy.mission[0])

    ''' Simplified mission reference '''
    m = jy.mission[0]
    print transects
    temp = []
    sal = []
    for t in transects:
        temp.extend(t['Temperature'].values)
        sal.extend(t['Salinity'].values)

    print np.nanmean(temp), np.nanstd(temp)
    print np.nanmean(sal), np.nanstd(sal)
    print sol.sol_SP_pt(np.nanmean(sal), np.nanmean(temp), gas='CH4', p_dry=1.86*1e-6, units='mM')/0.000001



    ''' Quick Transect Diagnosis '''
    # for i in range(6):
    #     plt.scatter(transects[i]['Longitude'], transects[i]['Latitude'], c=transects[i]['CH4_ppm'], cmap='coolwarm')
    #     plt.show()

    ####################################################
    ################ SPATIAL SLICES ####################
    ####################################################
    ''' Draws the path of the vehicle each day as distance from the boat launch '''
    targets = ('CH4_nM', 'CO2_uatm', 'O2Concentration', 'Salinity', 'Temperature')
    legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
                     'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
                     'O2Concentration': 'O$_2$ Concentration $\mu$M',
                     'Salinity':'Salinity, PSS',
                     'Temperature':'Temperature, C'}
    transect_labels = {3:'Transect D', 4:'Transect E', 0:'Transect A', 1:'Transect B', 2:'Transect C'}
    outfall_reference = (41.758022, -70.684128)

    # all_dist = []
    # num_samples = 0
    for m in transects[:-1]:
        # num_samples += len(m.index)
        # last_point = (m['Latitude'].values[0], m['Longitude'].values[0])
        # total_dist = 0
        m.loc[:, 'Distance'] = m.apply(lambda x: get_distance(outfall_reference, (x['Latitude'], x['Longitude'])), axis=1)
    #     for sample in range(len(m.index)):
    #         try:
    #             total_dist += np.fabs(get_distance(last_point, (m['Latitude'][sample], m['Longitude'][sample])))
    #         except:
    #             pass
    #         last_point = (m['Latitude'][sample], m['Longitude'][sample])
    #     all_dist.append(total_dist)
    # print all_dist
    # print np.sum(all_dist)
    # print np.mean(all_dist)
    # print np.std(all_dist)
    # print num_samples
    # asasd

    # get the plotting settings for the values
    vmin = []
    vmax = []
    rmin = 10000
    rmax = -10000
    for target in targets:
        temp_min = []
        temp_max = []
        for m in jy.mission:
            temp_min.append(np.nanmin(m[target].values))
            temp_max.append(np.nanmax(m[target].values))
        vmin.append(np.nanmin(temp_min))
        vmax.append(np.nanmax(temp_max))

    for m in transects[:-1]:
        rmin = np.nanmin([rmin, np.nanmin(m['Distance'].values)])
        rmax = np.nanmax([rmax, np.nanmax(m['Distance'].values)])

    # generate plots
    # transect_labels = ['Transect A', 'Transect B', 'Transect C', 'Transect D', 'Transect E']
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 8))
    #     for j, m in enumerate(transects[0:-1]):
    #         m = m.sort_values(by=target, ascending=True)
    #         scat = ax[j].scatter(m['Distance'], m['Depth'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         ax[j].axis([rmin-50.0, rmax+50.0, -0.1, 0.25])
    #         ax[j].axvline(0, 0, 10, c='r', linestyle='--')
    #         ax[j].set_title(transect_labels[j], fontsize=25)
    #         ax[j].set_aspect((rmax-rmin+100.)/(0.3))
    #     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.07, right=0.87, wspace=0.1)
    #     plt.gca().invert_yaxis()
    #     plt.gca().invert_xaxis()
    #     cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    #     cbar = fig.colorbar(scat, cax=cax)
    #     cbar.set_label(legend_labels[target], fontsize=24)
    #     plt.show()
    #     plt.close()

    ####################################################
    ################ SPATIAL REF MAP ###################
    ####################################################
    x_min = 1000
    x_max = -1000
    y_min = 1000
    y_max = -1000
    transect_labels = {3:'Transect D', 4:'Transect E', 0:'Transect A', 1:'Transect B', 2:'Transect C'}
    outfall_reference = (41.758022, -70.684128)

    for m in jy.mission:
        x_min = min(x_min, np.nanmin(m['Longitude']))
        y_max = max(y_max, np.nanmax(m['Latitude']))

        if np.nanmax(m['Longitude']) >= 0.0:
            pass
        else:
            x_max = max(x_max, np.nanmax(m['Longitude']))
        if np.nanmin(m['Latitude']) <= 0.0:
            pass
        else:
            y_min = min(y_min, np.nanmin(m['Latitude']))

    # base = Basemap(llcrnrlon=x_min-0.005, llcrnrlat=y_min-0.001, urcrnrlon=x_max+0.001, urcrnrlat=y_max+0.001,
    #                resolution='l', projection='cyl', suppress_ticks=False)

    # base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    # base.scatter(outfall_reference[1], outfall_reference[0], s=600, marker='*', label='Outfall', zorder=10, edgecolor='k', facecolor='r')
    # for radius in [150*i for i in range(10)]:
    #     lats, lons = getCircle(outfall_reference[0], outfall_reference[1], radius)
    #     base.plot(lons, lats, c='grey')
    #     if radius == 0:
    #         pass
    #     else:
    #         plt.gca().annotate(str(radius)+'m', xy=(lons[90], lats[90]+0.0001), fontsize=22, ha='center')

    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,5)), axis=0)
    # for i, m in enumerate(transects[0:-1]):
    #     base.scatter(m['Longitude'], m['Latitude'], label=transect_labels[i], s=1, c=colors[i], zorder=10-i, lw=0)

    # lgnd = plt.legend(loc='upper left')
    # for handle in lgnd.legendHandles[1:]:
    #     handle.set_sizes([200])

    # ax = plt.gca()
    # def xformat(x, pos=None): return lon2str(x)
    # def yformat(x, pos=None): return lat2str(x)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))
    # plt.gcf().subplots_adjust(bottom=0.04, top=0.98, left=0.15, right=0.98)

    # plt.show()
    # plt.close()


    ####################################################
    ################## SPATIAL MAPS ####################
    ####################################################
    ''' Draws the path of the vehicle each day as distance from the boat launch '''
    targets = ('CH4_nM', 'CO2_uatm', 'O2Concentration', 'Salinity', 'Temperature')
    legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
                     'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
                     'O2Concentration': 'O$_2$ Concentration $\mu$M',
                     'Salinity':'Salinity, PSS',
                     'Temperature':'Temperature, C'}
    transect_labels = {3:'Transect D', 4:'Transect E', 0:'Transect A', 1:'Transect B', 2:'Transect C'}
    outfall_reference = (41.758022, -70.684128)

    # get the plotting settings for the values
    vmin = []
    vmax = []
    rmin = 10000
    rmax = -10000
    for target in targets:
        temp_min = []
        temp_max = []
        for m in jy.mission:
            temp_min.append(np.nanmin(m[target].values))
            temp_max.append(np.nanmax(m[target].values))
        vmin.append(np.nanmin(temp_min))
        vmax.append(np.nanmax(temp_max))

    for m in transects[:-1]:
        rmin = np.nanmin([rmin, np.nanmin(m['Distance'].values)])
        rmax = np.nanmax([rmax, np.nanmax(m['Distance'].values)])

    # generate plots
    # transect_labels = ['Transect A', 'Transect B', 'Transect C', 'Transect D', 'Transect E']
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(20, 8))
    #     for j, m in enumerate(transects[:-1]):
    #         m = m.sort_values(by=target, ascending=True)
    #         base = Basemap(llcrnrlon=x_min-0.001, llcrnrlat=y_min-0.001, urcrnrlon=x_max+0.001, urcrnrlat=y_max+0.001,
    #                resolution='l', projection='cyl', suppress_ticks=False, ax=ax[j])

    #         base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    #         base.scatter(outfall_reference[1], outfall_reference[0], s=300, marker='*', label='Outfall', zorder=10, edgecolor='k', facecolor='r')
    #         scat=base.scatter(m['Longitude'], m['Latitude'], s=1, c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], rasterized=False)
    #         ax[j].set_title(transect_labels[j], fontsize=25)
    #         def xformat(x, pos=None): return lon2str(x)
    #         def yformat(x, pos=None): return lat2str(x)
    #         ax[j].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    #         ax[j].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))
    #     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.14, right=0.87, wspace=0.1)
    #     cax = fig.add_axes([0.89, 0.4, 0.01, 0.2])
    #     cbar = fig.colorbar(scat, cax=cax)
    #     cbar.set_label(legend_labels[target], fontsize=24)
    #     plt.show()
    #     plt.close()

    ####################################################
    #################### ST PLOTS ######################
    ####################################################
    salt = []
    temp = []
    for m in transects[:-1]:
        salt.extend(m['Salinity'].values)
        temp.extend(m['Temperature'].values)

    smin = np.nanmin(salt) - (0.01 * np.nanmax(salt))
    smax = np.nanmax(salt) + (0.01 * np.nanmax(salt))
    tmin = np.nanmin(temp) - (0.1 * np.nanmax(temp))
    tmax = np.nanmax(temp) + (0.1 * np.nanmax(temp))
    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1))
    ydim = int(round((tmax - tmin) / 0.1 + 1))
    # Create empty grid of zeros
    dens = np.zeros((ydim, xdim))
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1, ydim - 1, ydim) * 0.1 + tmin
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    # Loop to fill in grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.dens(si[i], ti[j], 0)
    # Substract 1000 to convert to sigma-t
    dens = dens - 1000

    # generate plots
    # transect_labels = ['Transect A', 'Transect B', 'Transect C', 'Transect D', 'Transect E']
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 8))
    #     for j, m in enumerate(transects[0:-1]):
    #         m = m.sort_values(by=target, ascending=True)
    #         scat = ax[j].scatter(m['Salinity'], m['Temperature'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         CS = ax[j].contour(si, ti, dens, linestyles='dashed', colors='grey')
    #         ax[j].clabel(CS, fontsize=12, inline=1, fmt='%1.0f')
    #         ax[j].axis([smin-0.05, smax+0.05, 24, 27])
    #         # ax[j].axvline(0, 0, 10, c='r', linestyle='--')
    #         ax[j].set_title(transect_labels[j], fontsize=25)
    #         ax[j].set_aspect(5)
    #     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.07, right=0.87, wspace=0.1)
    #     cax = fig.add_axes([0.89, 0.35, 0.01, 0.3])
    #     cbar = fig.colorbar(scat, cax=cax)
    #     cbar.set_label(legend_labels[target], fontsize=24)
    #     plt.show()
    #     plt.close()


    ####################################################
    ################ BARCHART TIMELINE #################
    ####################################################
    targets = ('CH4_nM', 'CO2_uatm', 'O2Concentration', 'Salinity', 'Temperature')
    legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
                     'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
                     'O2Concentration': 'O$_2$ Concentration $\mu$M',
                     'Salinity':'Salinity, PSS',
                     'Temperature':'Temperature, C'}
    transect_labels = {3:'Transect D', 4:'Transect E', 0:'Transect A', 1:'Transect B', 2:'Transect C'}
    outfall_reference = (41.758022, -70.684128)
    # transects = ['Transect A', 'Transect B', 'Transect C', 'Transect D', 'Transect E']
    distance_inc = 50.

    avgs = []
    stds = []
    num_dists = int(np.ceil((rmax--700)/distance_inc))
    width = 0.04
    dists = [-700+i*distance_inc for i in range(0, num_dists)]
    colors = plt.cm.Spectral(np.linspace(0,1,len(dists)))
    last_dist = round(rmin, -1)
    ind = np.arange(len(transects[:-1]))


    # for target in targets:
    #     last_dist = -700#round(rmin, -2)
    #     top_trend = []
    #     mid_trend = []
    #     bottom_trend = []
    #     for m in transects[:-1]:
    #         top_trend.append(np.mean(m[(m['Distance'] <= 25.0) & (m['Distance'] > -25.0)][target]))
    #         mid_trend.append(np.mean(m[(m['Distance'] <= 475.) & (m['Distance'] > 425.)][target]))
    #         bottom_trend.append(np.mean(m[(m['Distance'] <= -225.) & (m['Distance'] > -275.)][target]))

    #     plt.figure(figsize=(15,3))
    #     for i, d in enumerate(dists[1:]):
    #         for j, m in enumerate(transects[:-1]):
    #             tmp = m[(m['Distance'] <= d) & (m['Distance'] > last_dist)]
    #             avgs.append(np.mean(tmp[target]))
    #             stds.append(np.std(tmp[target]))
    #         # plt.plot(dates, avgs, c=colors[i], label=str(last_depth) + 'm-' + str(d) + 'm')
    #         # plt.errorbar(dates, avgs, yerr=stds, c=colors[i], label=str(last_depth) + 'm-' + str(d) + 'm')
    #         # plt.fill_between(dates, [a-s for a,s in zip(avgs,stds)], [a+s for a,s in zip(avgs, stds)], alpha=0.1, color=colors[i])
    #         plt.bar(ind + (width*i - (num_dists*width/2)),
    #                 avgs,
    #                 yerr=stds,
    #                 color=colors[i],
    #                 width=width,
    #                 label=str(last_dist) + ' : ' + str(d) + 'm',
    #                 error_kw={'ecolor':'red', 'elinewidth':0.5})
    #         last_dist = d
    #         avgs = []
    #         stds = []
    #     plt.plot(ind + (width*14-(num_dists*width/2)), top_trend, c=colors[14], marker='o', lw=3, linestyle='--', mec='k', ms=10)
    #     plt.plot(ind + (width*21-(num_dists*width/2)), mid_trend, c=colors[21], marker='o', lw=3, linestyle='--', mec='k', ms=10)
    #     plt.plot(ind + (width*9-(num_dists*width/2)), bottom_trend, c=colors[9], marker='o', lw=3, linestyle='--', mec='k', ms=10)

    #     plt.xlabel('Transect', fontsize=16)
    #     plt.gca().set_xticks(ind)
    #     # plt.gca().set_xticklabels(transects, fontsize=14)
    #     plt.ylabel(legend_labels[target])
    #     box = plt.gca().get_position()
    #     plt.gca().set_position([box.x0, box.y0, box.width*0.8, box.height])
    #     # plt.gca().set_aspect((rmax-rmin+100.)/(5.1))
    #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    #     plt.show()
    #     plt.close()

    ####################################################
    ################ PROPERTIES COMPARE ################
    ####################################################
    plt.scatter(m['O2Concentration'], m['CH4_nM'])
    plt.show()

    plt.scatter(m['Salinity'], m['O2Concentration'], c=m['CH4_nM'], cmap='viridis')
    plt.colobar()
    plt.show()

    plt.scatter(m['O2Concentration'], m['CO2_uatm'])
    plt.show()

