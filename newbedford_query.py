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
        return np.sqrt(dist) * np.sign(coord2[0]-coord1[0])
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

    base_path = '/home/vpreston/Documents/IPP/nb-effluent-plumes/'
    ctd_dirs = [base_path + 'data/ctd/ctd_data.txt']
    gga_dirs = [base_path + 'data/gga/gga_329_data.txt']
    op_dirs = [base_path + 'data/op/optode_20180329181123.txt',
               base_path + 'data/op/optode_20180329192656.txt',
               base_path + 'data/op/optode_20180329204400.txt',
               base_path + 'data/op/optode_20180329211740.txt',
               base_path + 'data/op/optode_20180329213909.txt',
               base_path + 'data/op/optode_20180329223353.txt',
               base_path + 'data/op/optode_20180329230511.txt']
    airmar_dirs = [base_path + 'data/airmar/airmar_20180329181245.txt',
                   base_path + 'data/airmar/airmar_20180329191141.txt',
                   base_path + 'data/airmar/airmar_20180329204336.txt',
                   base_path + 'data/airmar/airmar_20180329213838.txt',
                   base_path + 'data/airmar/airmar_20180329221731.txt',
                   base_path + 'data/airmar/airmar_20180329230448.txt']
    pix_dirs = [base_path + 'data/pix/43.log.gpx']
    sonde_dirs = [base_path + 'data/sonde/sonde.csv']

    mission_name = 'newbed_2018.csv'

    trim_values = None
    bounds = [2458207+0.055, 2458207+0.274]
    offset =  2440587.50375
    gga_offset = -0.002

    ####################################################
    ############ PLOTTING DEFAULTS  ####################
    ####################################################
    matplotlib.rcParams['figure.figsize'] = (15,15)
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['figure.titlesize'] = 15
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.labelsize'] = 15
    matplotlib.rcParams['legend.fontsize'] = 15
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['savefig.directory'] = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/newbed/'


    ####################################################
    ################# PROCESS DATA #####################
    ####################################################
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset, gga_offset])
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # jy.attach_sensor('optode', op_dirs)
    # jy.attach_sensor('sonde', sonde_dirs)
    # jy.attach_sensor('pixhawk', pix_dirs)

    # # Can now perform work with the sensors
    # jy.create_mission({'geoframe':'pixhawk'})

    # # Check everything
    # m = jy.mission[0]
    # # print m.head(10)

    # plt.plot(jy.ctd.get_df()['Julian_Date'], jy.ctd.get_df()['Salinity']/np.nanmax(jy.ctd.get_df()['Salinity']))
    # plt.plot(jy.gga.get_df()['Julian_Date'], jy.gga.get_df()['CO2_ppm']/np.nanmax(jy.gga.get_df()['CO2_ppm']))
    # plt.plot(jy.optode.get_df()['Julian_Date'], jy.optode.get_df()['O2Concentration']/np.nanmax(jy.optode.get_df()['O2Concentration']))
    # plt.show()

    # plt.plot(jy.pixhawk.get_df()['Longitude'], jy.pixhawk.get_df()['Latitude'])
    # plt.show()


    # # Create Zones
    # zones = [m[m.index < 2458207+0.206],
    #          m[(m.index > 2458207+0.228)]]

    # plt.plot(zones[0]['pixhawk']['Longitude'], zones[0]['pixhawk']['Latitude'])
    # plt.plot(zones[1]['pixhawk']['Longitude'], zones[1]['pixhawk']['Latitude'])
    # plt.show()

    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/newbed/', mission_name=mission_name)
    
    # for i, z in enumerate(zones):
    #     z.to_csv('/home/vpreston/Documents/IPP/jetyak_parsing/missions/newbed/zone_'+str(i)+'.csv')

    ####################################################
    ####### READ IN PRE-PROCESSED DATA #################
    ####################################################
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/newbed/'
    miss = [mission_name]
    titles = ['NewBedford 2018']

    ''' Create mission operator '''
    # jy = jetyak.JetYak()

    ''' If the file isn't simlified or conversions made, run this '''
    # jy.load_mission([base_path+m for m in miss], header=[0,1], simplify_mission=True, meth_eff=0.15, carb_eff=0.70)
    # jy.save_mission(base_path, 'trimmed_chemyak_cleaned')

    # for i in [0, 1]:
    #     jy = jetyak.JetYak()
    #     jy.load_mission([base_path+'zone_'+str(i)+'.csv'], header=[0,1], simplify_mission=True, meth_eff=0.15, carb_eff=0.70)
    #     jy.save_mission(base_path, 'trimmed_zone_'+str(i))

    ''' Read in simplified targets'''
    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_chemyak_cleaned_'+str(i)+'.csv' for i in [0]], header=0, simplify_mission=False)

    zones = []
    for i in [0, 1]:
        temp_jy = jetyak.JetYak()
        temp_jy.load_mission([base_path+'trimmed_zone_'+str(i)+'_0.csv'], header=0, simplify_mission=False)
        m = temp_jy.mission[0]
        m = m[m['Depth'] > 0.25]
        zones.append(m)

    m = zones
    plt.plot(m[0]['Longitude'], m[0]['Latitude'])
    plt.plot(m[1]['Longitude'], m[1]['Latitude'])
    plt.show()

    plt.figure()
    plt.scatter(m[0].index, m[0]['Depth'], c=m[0]['Salinity'], cmap='viridis', vmin=25, vmax=30)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Salinity')
    plt.figure()
    plt.scatter(m[0].index, m[0]['Depth'], c=m[0]['Temperature'], cmap='viridis', vmin=4.8, vmax=5.3)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Temperature')
    plt.figure()
    plt.scatter(m[0].index, m[0]['Depth'], c=m[0]['CH4_nM'], cmap='viridis', vmin=0, vmax=20)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Methane')
    plt.show()

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
    ascent_direction = [True, True, False, False, True]
    zone_labels = {0:'Inner Harbor', 1:'Outer Harbor'}

    outfall_reference = [(41.63118, -70.90668), (41.58497, -70.89417)]

    # Get the distance from the outfall reference, where negative implies "under" the outfall, in terms of latitude
    for i, m in enumerate(zones):
        m.loc[:, 'Distance'] = m.apply(lambda x: (get_distance(outfall_reference[i], (x['Latitude'], x['Longitude']))), axis=1)

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

    for m in zones:
        rmin = np.nanmin([rmin, np.nanmin(m['Distance'].values)])
        rmax = np.nanmax([rmax, np.nanmax(m['Distance'].values)])

    # generate plots
    # zone_labels = ['Inner Harbor', 'Outer Harbor']
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, len(zones), sharex=True, sharey=True, figsize=(15, 8))
    #     for j, m in enumerate(zones):
    #         m = m.sort_values(by=target, ascending=ascent_direction[j])
    #         scat = ax[j].scatter(m['Distance'], m['Depth'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         ax[j].axis([rmin-25.0, rmax+25.0, -0.1, 10.0])
    #         ax[j].axvline(0, 0, 10, c='r', linestyle='--')
    #         ax[j].set_title(zone_labels[j], fontsize=25)
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
    zone_labels = {0:'Inner Harbor', 1:'Outer Harbor'}   

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
    # base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose=True)

    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,5)), axis=0)

    # for i, m in enumerate(zones):
    #     base.scatter(outfall_reference[i][1], outfall_reference[i][0], s=600, marker='*', label='Outfall', zorder=10, edgecolor='k', facecolor='r')
    #     base.scatter(m['Longitude'], m['Latitude'], label=zone_labels[i], s=1, c=colors[i], zorder=9-i, lw=0)

    # ax = plt.gca()
    # def xformat(x, pos=None): return lon2str(x)
    # def yformat(x, pos=None): return lat2str(x)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))
    # plt.gcf().subplots_adjust(bottom=0.04, top=0.98, left=0.15, right=0.98)

    # plt.show()
    # plt.close()

    # # Do for individual sites
    # for i, m in enumerate(zones):
    #     base = Basemap(llcrnrlon=np.nanmin(m['Longitude'])-0.001,
    #                    llcrnrlat=np.nanmin(m['Latitude'])-0.001,
    #                    urcrnrlon=np.nanmax(m['Longitude'])+0.001,
    #                    urcrnrlat=np.nanmax(m['Latitude'])+0.001,
    #                    resolution='l', projection='cyl', suppress_ticks=False)
    #     base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose=True)

    #     base.scatter(outfall_reference[i][1], outfall_reference[i][0], s=600, marker='*', label='Outfall', zorder=10, edgecolor='k', facecolor='r')
    #     base.scatter(m['Longitude'], m['Latitude'], label=zone_labels[i], s=10, c='k', zorder=9-i, lw=0)
    #     plt.show()
    #     plt.close()


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

    for m in zones:
        rmin = np.nanmin([rmin, np.nanmin(m['Distance'].values)])
        rmax = np.nanmax([rmax, np.nanmax(m['Distance'].values)])

    # generate plots
    # zone_labels = ['Inner Harbor', 'Outer Harbor']
    
    # for j, m in enumerate(zones):
    #     fig, ax = plt.subplots(1, len(targets), sharex=True, sharey=True, figsize=(20, 8))
    #     for i, target in enumerate(targets):
    #         m = m.sort_values(by=target, ascending=ascent_direction[j])
    #         base = Basemap(llcrnrlon=np.nanmin(m['Longitude'])-0.001,
    #                        llcrnrlat=np.nanmin(m['Latitude'])-0.001,
    #                        urcrnrlon=np.nanmax(m['Longitude'])+0.001,
    #                        urcrnrlat=np.nanmax(m['Latitude'])+0.001,
    #                        resolution='l', projection='cyl', suppress_ticks=False, ax=ax[i])
    #         # base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    #         base.scatter(outfall_reference[j][1], outfall_reference[j][0], s=300, marker='*', label='Outfall', zorder=10, edgecolor='k', facecolor='r')
    #         scat=base.scatter(m['Longitude'], m['Latitude'], s=1, c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], rasterized=False)
    #         ax[i].set_title(legend_labels[target], fontsize=15)
    #         def xformat(x, pos=None): return lon2str(x)
    #         def yformat(x, pos=None): return lat2str(x)
    #         ax[i].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    #         ax[i].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))
    #         plt.colorbar(scat, ax=ax[i], shrink=0.5)
    #         plt.subplots_adjust(hspace = 100)
    #     plt.show()
    #     plt.close()

    ####################################################
    #################### ST PLOTS ######################
    ####################################################
    salt = []
    temp = []
    for m in zones:
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
    # zone_labels = ['Inner Harbor', 'Outer Harbor']
    # for j, m in enumerate(zones):
    #     fig, ax = plt.subplots(1, len(targets)-2, sharex=True, sharey=True, figsize=(15, 8))
    #     for i, target in enumerate(targets[:-2]):
    #         m = m.sort_values(by=target, ascending=True)
    #         scat = ax[i].scatter(m['Salinity'], m['Temperature'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         CS = ax[i].contour(si, ti, dens, linestyles='dashed', colors='grey')
    #         ax[i].clabel(CS, fontsize=12, inline=1, fmt='%1.0f')
    #         ax[i].set_title(legend_labels[target], fontsize=15)
    #         ax[i].set_aspect(5)
    #         plt.colorbar(scat, ax=ax[i], shrink=0.75)
    #         plt.subplots_adjust(hspace = 150)
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
    distance_inc = 25.

    avgs = []
    stds = []
    num_dists = int(np.ceil((rmax--700)/distance_inc))
    width = 25.
    dists = [-700+i*distance_inc for i in range(0, num_dists)]
    colors = plt.cm.Spectral(np.linspace(0,1,len(dists)))
    last_dist = round(rmin, -1)
    ind = np.arange(len(zones))


    for target in targets:
        last_dist = -700#round(rmin, -2)
        for j, m in enumerate(zones):
            plt.figure(figsize=(15,3))
            for i, d in enumerate(dists[1:]):
                tmp = m[(m['Distance'] <= d) & (m['Disatnce'] > last_dist)]
                avgs.append(np.mean(tmp[target]) - np.mean(m[target]))
                # stds.append(np.std(tmp[target]))
                plt.bar(d,
                        avgs,
                        # yerr=stds,
                        color='b',#colors[i],
                        width=width,
                        label=str(last_dist) + ' : ' + str(d) + 'm',
                        error_kw={'ecolor':'red', 'elinewidth':0.5})
                last_dist = d
                avgs = []
                stds = []

            plt.xlabel('Zone', fontsize=16)
            plt.ylabel(legend_labels[target])
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width*0.8, box.height])
            # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
            plt.show()
            plt.close()

    ####################################################
    ################ PROPERTIES COMPARE ################
    ####################################################
    # plt.scatter(m['O2Concentration'], m['CH4_nM'])
    # plt.show()

    # plt.scatter(m['O2Concentration'], m['CO2_uatm'])
    # plt.show()

