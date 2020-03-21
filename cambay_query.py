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
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as mb
from scipy import stats

def process_wind(filename):
    ''' Helper method to process wind data from Cambridge Bay deployment '''
    df = pd.read_table(filename, delimiter=',', header=0, engine='c')
    df.loc[:, 'Minute'] = 0.0
    df.loc[:, 'Hour'] = df['Time'].str.split(':').str.get(0).astype('float')+6.0
    df.loc[:, 'Second'] = 0.0
    df = sensors.make_global_time(df)
    return df

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

    ''' June 28th, 2018 '''
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.28.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # gga_dirs = [base_path + 'gga/2018-06-28/gga_2018-06-28_f0001.txt']
    # op_dirs = [base_path + 'op/optode_20180330013534.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330013612.txt']
    # mission_name = '0628_interp.csv'
    # trim_values = [[2458298.3855, 2458298.3845], [2458298.394, 2458298.3925], [2458298.422, 2458298.42]]
    # bounds = [2458298.339353009, 2458298.4214641205]
    # offset = 2440678.4842

    ''' June 29th, 2018 '''
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.29.2018/data/'
    # ctd_dirs = [base_path+'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path+'airmar/airmar_20180330034652.txt', base_path+'airmar/airmar_20180330082958.txt']
    # gga_dirs = [base_path+'gga/2018-06-29/gga_2018-06-29_f0002.txt']
    # op_dirs = [base_path + 'op/optode_20180330034739.txt', base_path + 'op/optode_20180330082905.txt']
    # mission_name = '0629_interp.csv'
    # trim_values = [[2458299.3626, 2458299.362]]
    # bounds = [2458299.200787037, 2458299.410834491]
    # offset = 2440679.255

    ''' June 30th, 2018 '''
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.30.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330093911.txt']
    # gga_dirs = [base_path + 'gga/2018-06-30/gga_2018-06-30_f0001.txt']
    # op_dirs = [base_path + 'op/optode_20180330093807.txt']
    # mission_name = '0630_interp.csv'
    # trim_values = [[2458300.382, 2458300.3813]]
    # bounds = [2458300.3061458333, 2458300.456943287]
    # offset = 2440680.11575

    ''' July 1, 2018 '''
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.01.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330132521.txt']
    # gga_dirs = [base_path + 'gga/2018-07-01/gga_2018-07-01_f0001.txt']
    # op_dirs = [base_path + 'op/optode_20180330132615.txt']
    # mission_name = '0701_interp.csv'
    # trim_values = None
    # bounds = [2458301.2410046295, 2458301.4014780093]
    # offset = 2440680.893

    ''' July 2, 2018 '''
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.02.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330172728.txt']
    # gga_dirs = [base_path + 'gga/2018-07-02/gga_2018-07-02_f0001.txt', base_path + 'gga/2018-07-02/gga_2018-07-02_f0002.txt']
    # op_dirs = [base_path + 'op/optode_20180330172646.txt']
    # mission_name = '0702_interp.csv'
    # trim_values = [[2458302.186, 2458302.184]]
    # bounds = [2458302.171130787, 2458302.3713055556]
    # offset = 2440681.650

    ''' July 4, 2018 '''
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.04.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330223042.txt']
    # gga_dirs = [base_path + 'gga/2018-07-04/gga_2018-07-04_f0003.txt', base_path + 'gga/2018-07-04/gga_2018-07-04_f0004.txt']
    # op_dirs = [base_path + 'op/optode_20180330223020.txt']
    # mission_name = '0704_interp.csv'
    # trim_values = [[2458304.39, 2458304.352],[2458304.3085, 2458304.3077],[2458304.256, 2458304.2547],
    #               [2458304.304, 2458304.303],[2458304.3095,2458304.308], [2458304.347, 2458304.345]]
    # bounds = [2458304.2128425925, 2458304.345]
    # offset = 2440683.486

    ####################################################
    ############ PLOTTING DEFAULTS  ####################
    ####################################################
    matplotlib.rcParams['figure.figsize'] = (15,15)
    matplotlib.rcParams['font.size'] = 24
    matplotlib.rcParams['figure.titlesize'] = 28
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.labelsize'] = 24
    matplotlib.rcParams['legend.fontsize'] = 22
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['savefig.directory'] = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/cambay/'


    ####################################################
    ################# PROCESS DATA #####################
    ####################################################
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset])
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # jy.attach_sensor('optode', op_dirs)
    # # # print np.sort(jy.airmar.get_df()['Julian_Date'].values)[0], np.sort(jy.airmar.get_df()['Julian_Date'].values)[-1]

    # # # Can now perform work with the sensors
    # # jy.create_mission({'geoframe':'airmar'})
    # jy.create_mission({'geoframe':'gga'})
    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/cambay/', mission_name=mission_name)

    ####################################################
    ####### READ IN PRE-PROCESSED DATA #################
    ####################################################
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/cambay/'
    miss = ['0628.csv', '0629.csv', '0630.csv',
            '0701.csv', '0702.csv', '0704.csv']
    titles = ['June 28', 'June 29', 'June 30', 'July 1', 'July 2', 'July 4']

    ''' Create mission operator '''
    jy = jetyak.JetYak()

    ''' If the file isn't simlified or conversions made, run this '''
    # jy.load_mission([base_path+m for m in miss], header=[0,1], simplify_mission=True, meth_eff=0.0509, carb_eff=0.505)
    # jy.save_mission(base_path, 'trimmed_chemyak_cleaned')
    # jy.save_mission(base_path, 'trimmed_ch4_0509_co2_505')

    ''' Read in simplified targets and bottle sample files '''
    jy.load_mission([base_path+'trimmed_ch4_0509_co2_505_'+str(i)+'.csv' for i in [0,1,2,3,4,5]], header=0, simplify_mission=False)
    # jy.load_mission([base_path+'trimmed_chemyak_cleaned_'+str(i)+'.csv' for i in [0,1,2,3,4,5]], header=0, simplify_mission=False)
    jy.add_bottle_samples('/home/vpreston/Documents/IPP/cb-methane-data/discrete_samples.csv')
    

    ####################################################
    ############### WINDSPEED ANALYSIS #################
    ####################################################
    ''' Note: Will work with unsimplified mission files '''
    # filepath = '/home/vpreston/Documents/IPP/jetyak_parsing/airport_wind.csv'
    # wind = process_wind(filepath)

    # for i in range(0,len(miss)):
    #     wind_speed = jy.mission[i]['airmar']['wind_speed_M'].values * (10./1.5) ** 0.11
    #     #get wind values
    #     times = jy.mission[i]['Unnamed: 0_level_0']['Unnamed: 0_level_1'].values[1:]
    #     times = [float(j) for j in times]
    #     valid_wind = wind[(wind['Julian_Date'] >= np.nanmin(times)) & (wind['Julian_Date'] <= np.nanmax(times))]
    #     plt.scatter(times, wind_speed[1:], c='k', alpha=0.5, s=1)
    #     plt.scatter(valid_wind['Julian_Date'], valid_wind['Wind Spd (km/h)'].values*0.277778, c='r', s=100)
    #     plt.xlabel('Time (Julian Date)')
    #     plt.ylabel('Wind Speed (10m Elevation, m/s)')
    #     plt.title(titles[i])
    #     plt.show()
    #     plt.close()

    ####################################################
    ################ CTD CAST ANALYSIS #################
    ####################################################
    ''' Targets data from July 1st and a given coordinate to compare against a CTD cast '''
    # cast = jy.mission[3]
    # cast = cast[(np.fabs(cast['Latitude']-69.105) < 0.01) & (np.fabs(cast['Longitude']--105.0417) < 0.01)]
    # np.savetxt('temp.csv', cast['Depth'].values)
    # np.savetxt('tempsal.csv', cast['Salinity'].values)

    for m in jy.mission:
    	print np.nanmin(m['CH4_nM'])
    	print np.nanmax(m['CH4_nM'])
    	print '----'

    ####################################################
    ################ BOTTLE SAMPLES ####################
    ####################################################
    # targets = (('co2_mean', 'jco2_mean', 'co2_std', 'jco2_std'),
    #            ('ch4_mean', 'jch4_mean', 'ch4_std', 'jch4_std'),
    #            ('ch4_mean', 'co2_mean', 'ch4_std', 'co2_std'))
    # legend_labels = {1:'01 July', 2:'02 July', 28:'28 June', 29:'29 June', 30:'30 June'}
    collapsed = jy.collapse_bottle_samples() #collapse replicates in the bottle sample dataset
    matched = jy.match_bottles(collapsed, geo_epsilon=50., depth_epsilon=0.25) #find matching ChemYak data
    matched = matched.dropna()
    def right_std(sal):
        if sal > 10:
            return 13
        else:
            return 150
    matched.loc[:, 'co2_std'] = matched.apply(lambda x: right_std(x['jsal_mean']), axis=1) #add the pco2 error reported by Patrick
    matched.to_csv(base_path+'agg_bottle.csv')
    # groups = matched.groupby('day') #group the data by day in order to plot
    # c = np.flip(plt.cm.viridis(np.linspace(0,1,5)), axis=0)
    # colors = {28: c[0], 29: c[1], 30: c[2], 1:c[3], 2:c[4]}

    # for target in targets:
    #     for name, group in groups:
    #         if float(name) == 2: #remove an outlier from July 2nd dataset
    #             group = group[group['ch4_mean'] < 300]
    #         plt.errorbar(group[target[0]], group[target[1]],
    #                      xerr=group[target[2]], yerr=group[target[3]],
    #                      fmt='o', label=legend_labels[name], color=colors[name],
    #                      markersize=20, elinewidth=2.0, mew=3, mfc='none')

    #     matched = matched[((matched['day']==2) & (matched['ch4_mean']<300))|(matched['day'] != 2)]
    #     ax_lim = np.linspace(0, np.nanmax(matched[target[0]])+50, 10)
    #     # plt.plot(ax_lim, ax_lim, 'k', label='')
    #     mask = ~np.isnan(matched[target[0]]) & ~np.isnan(matched[target[1]])
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(matched[target[0]].values[mask],
    #                                                                    matched[target[1]].values[mask])
    #     print target
    #     print 'Slope: ', slope, ' Intercept: ', intercept, ' Rvalue: ', r_value
    #     # plt.plot(ax_lim, slope*ax_lim+intercept, 'r--', label='Best Fit Line')
    #     x = matched[target[0]].values[:, np.newaxis]
    #     a, _, _, _ = np.linalg.lstsq(x, matched[target[1]].values[mask])
    #     ss_tot = np.sum([(x-np.nanmean(matched[target[1]]))**2 for x in matched[target[1]].values])
    #     ss_res = np.sum([(y-x*a)**2 for y, x, in zip(matched[target[1]].values, matched[target[0]].values)])
    #     r2 = 1 - ss_res/ss_tot
    #     print 'Forced Slope: ', a, ' R2 Value: ', r2
    #     print '-----'
    #     plt.plot(ax_lim, a*ax_lim, 'k--', label='Line of Fit', zorder=10)
    #     plt.axis('square')
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = [0, 3, 4, 5, 1, 2]
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    #     # plt.legend()

    #     plt.gcf().text(0.75, 0.19, 'Slope = '+str(np.round(a[0], 3)), ha='right')
    #     plt.gcf().text(0.75, 0.15, '$R^2$ Value = '+str(np.round(r2,3)), ha='right')

    #     if 'co2' in target[0]:
    #         plt.xlabel(r'Bottle Sample CO$_2$ ($\mu$atm)')
    #     else:
    #         plt.xlabel('Bottle Sample CH$_4$ (nM)')

    #     if 'jco2' in target[1]:
    #         plt.ylabel(r'ChemYak Observered CO$_2$ ($\mu$atm)')
    #     elif 'jch4' in target[1]:
    #         plt.ylabel('ChemYak Observed CH$_4$ (nM)')
    #     else:
    #         plt.ylabel(r'Bottle Sample CO$_2$ ($\mu$atm)')

    #     # plt.gcf().tight_layout(rect=(0,0.01,1,1))
    #     plt.show()
    #     plt.close()

    ####################################################
    ################ SPATIAL EXPLORATION ###############
    ####################################################
    # for m in jy.mission:
    #     scat = plt.scatter(m['CH4_nM'], m['CO2_uatm'], alpha=0.1, c=m['Latitude'], cmap='viridis')
    #     plt.xlabel('Methane, nM')
    #     plt.ylabel('pCO2, uatm')
    #     cbar = plt.colorbar(scat)
    #     cbar.set_label('Latitude')
    #     plt.show()
    #     plt.close()

    # for m in jy.mission:
    #     scat = plt.scatter(m['Longitude'], m['Depth'], c=(m['Temperature']), cmap='viridis', vmin=4.5, vmax=6)
    #     plt.xlabel('Longitude')
    #     plt.ylabel('Depth')
    #     plt.gca().invert_yaxis()
    #     cbar = plt.colorbar(scat)
    #     cbar.set_label('Temeprature, C')
    #     plt.show()
    #     plt.close()

    for m in jy.mission:
        temp = m[m['Depth'] <= 1.0]
        print np.mean(temp.Salinity.values), np.std(temp.Salinity.values)
    ####################################################
    ################ SPATIAL SLICES ####################
    ####################################################
    ''' Draws the path of the vehicle each day as distance from the boat launch '''
    targets = ('CH4_nM', 'CO2_uatm', 'Salinity', 'Temperature')
    legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
                     'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
                     'Salinity':'Salinity, PSS',
                     'Temperature':'Temperature, C'}
    date_labels = {3:'July 01, 2018', 4:'July 02, 2018', 0:'June 28, 2018', 1:'June 29, 2018', 2:'June 30, 2018'}
    dock_reference = (69.121595, -105.019215)
    dock_reference = (69.125566, -105.004188)

    # all_dist = []
    # num_samples = 0
    for m in jy.mission:
    #     num_samples += len(m.index)
    #     last_point = (m['Latitude'].values[0], m['Longitude'].values[0])
    #     total_dist = 0
        m.loc[:, 'Distance'] = m.apply(lambda x: get_distance(dock_reference, (x['Latitude'], x['Longitude'])), axis=1)
    #     for sample in range(len(m.index)):
    #         try:
    #             total_dist += np.fabs(get_distance(last_point, (m['Latitude'][sample], m['Longitude'][sample])))
    #         except:
    #             pass
    #         last_point = (m['Latitude'][sample], m['Longitude'][sample])
    #     all_dist.append(total_dist)
    # print all_dist
    # print np.mean(all_dist)
    # print np.std(all_dist)
    # print num_samples
    # asasd

    #june 29, embayment
    temp = jy.mission[1]
    temp = temp[temp['Depth'] < 2]
    embayment = temp[temp['Distance'] < 0]
    mouth = temp[(temp['Distance'] < 100) & (temp['Distance'] > 0)]
    print 'Embayment, CH4: ', np.mean(embayment['CH4_nM'].values), np.std(embayment['CH4_nM'].values)
    print 'Mouth, CH4: ', np.mean(mouth['CH4_nM'].values), np.std(mouth['CH4_nM'].values)

    print 'Embayment, CO2: ', np.mean(embayment['CO2_uatm'].values), np.std(embayment['CO2_uatm'].values)
    print 'Mouth, CO2: ', np.mean(mouth['CO2_uatm'].values), np.std(mouth['CO2_uatm'].values)


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

    for m in jy.mission:
        rmin = np.nanmin([rmin, np.nanmin(m['Distance'].values)])
        rmax = np.nanmax([rmax, np.nanmax(m['Distance'].values)])

    # # generate plots
    # date_labels = ['29 June', '2 July']
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 8))
    #     for j, m in enumerate([jy.mission[1], jy.mission[4]]):
    #         m = m.sort_values(by=target, ascending=True)
    #         scat = ax[j].scatter(m['Distance'], m['Depth'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         ax[j].axis([rmin-50.0, rmax+50.0, -0.1, 6])
    #         ax[j].axvline(0, 0, 10, c='r', linestyle='--')
    #         # ax[j].annotate('Towards Receding Ice Edge', xy=(rmin, 9.8), ha='left', va='bottom', color='grey', rotation=90.)
    #         # ax[j].annotate('Towards Freshwater Creek', xy=(rmax, 9.8), ha='right', va='bottom', color='grey', rotation=270.)
    #         ax[j].set_title(date_labels[j], fontsize=36)
    #         ax[j].set_aspect((rmax-rmin+100.)/(6.1))
    #     fig.text(0.45, 0.04, 'Distance from Freshwater Creek Mouth, m', va='center', ha='center')
    #     fig.text(0.03, 0.5, 'Depth, m', va='center', ha='center', rotation='vertical')
    #     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.07, right=0.87, wspace=0.1)
    #     plt.gca().invert_yaxis()
    #     plt.gca().invert_xaxis()
    #     cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    #     cbar = fig.colorbar(scat, cax=cax)
    #     cbar.set_label(legend_labels[target], fontsize=24)
    #     plt.show()
    #     plt.close()

    # date_labels = ['28 June', '29 June', '30 June', '01 July', '02 July']
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(18, 8), gridspec_kw=dict(wspace=0.1, left=0.05, bottom=0.1))
    #     for j, m in enumerate(jy.mission[:-1]):
    #         m = m.sort_values(by=target, ascending=True)
    #         scat = ax[j].scatter(m['Distance'], m['Depth'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         ax[j].axis([rmin-50.0, rmax+50.0, -0.1, 6])
    #         ax[j].axvline(0, 0, 10, c='r', linestyle='--')
    #         # ax[j].annotate('Towards Receding Ice Edge', xy=(rmin, 9.8), ha='left', va='bottom', color='grey', rotation=90.)
    #         # ax[j].annotate('Towards Freshwater Creek', xy=(rmax, 9.8), ha='right', va='bottom', color='grey', rotation=270.)
    #         ax[j].set_title(date_labels[j])
    #         ax[j].set_aspect((rmax-rmin+100.)/(6.1))
    #     fig.text(0.45, 0.2, 'Distance from Freshwater Creek Mouth, m', va='center', ha='center')
    #     fig.text(0.02, 0.5, 'Depth, m', va='center', ha='center', rotation='vertical')
    #     plt.gca().invert_yaxis()
    #     plt.gca().invert_xaxis()
    #     plt.xticks([0, 1250, 2500])
    #     cax = fig.add_axes([0.91, 0.25, 0.01, 0.45])
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
    dock_reference = (69.124361, -105.011098)
    dock_reference = (69.125566, -105.004188)
    date_labels = {3:'01 July', 4:'02 July', 0:'28 June', 1:'29 June', 2:'30 June'}

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

    # base = Basemap(llcrnrlon=x_min-0.01, llcrnrlat=y_min-0.005, urcrnrlon=x_max+0.005, urcrnrlat=y_max+0.01,
    #                resolution='l', projection='cyl', suppress_ticks=False)
    base = Basemap(urcrnrlon=x_min-0.015, llcrnrlat=y_min-0.01, llcrnrlon=x_max+0.015, urcrnrlat=y_max+0.01,
                   resolution='c', suppress_ticks=True, epsg=6125)
    
    # base = Basemap(width=3000,height=4250,
    #         rsphere=(6378137.00,6356752.3142),\
    #         resolution='f' ,area_thresh=10.,projection='lcc',\
    #         lat_1=y_min-0.05,lat_2=y_max+0.05,lat_0=(y_min+y_max)/2,lon_0=(x_min+x_max)/2)
    base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    # base.drawlsmask(land_color='coral', ocean_color='aqua', lakes=True)
    # base.drawcoastlines()
    # base.drawrivers()
    # base.fillcontinents(color='coral',lake_color='aqua')
    # base.drawmapboundary(fill_color='aqua')
    dx, dy = base(dock_reference[1], dock_reference[0])
    base.scatter(dx, dy, s=500, marker='*', label='Freshwater Creek Mouth', zorder=10, edgecolor='k', facecolor='r')
    for radius in [500*i for i in range(10)]:
        lats, lons = getCircle(dock_reference[0], dock_reference[1], radius)
        lons, lats = base(lons, lats)
        base.plot(lons, lats, c='grey')
        if radius == 0:
            pass
            # plt.gca().annotate('Embayment', xy=(lons[270], lats[270]+0.001), xytext=(lons[270]+0.0005, lats[270]+0.002), fontsize=22, ha='center')
            # plt.gca().annotate('Freshwater Creek Mouth', xy=(lons[270], lats[270]+0.0005), fontsize=10, ha='right')
        else:
            plt.gca().annotate(str(radius)+'m', xy=(lons[270], lats[270]+0.0003), fontsize=22, ha='center')

    colors = np.flip(plt.cm.viridis(np.linspace(0,1,5)), axis=0)
    for i, m in enumerate(jy.mission[0:5]):
        mx, my = base(m['Longitude'].values, m['Latitude'].values)
        base.scatter(mx, my, label=date_labels[i], s=1, c=colors[i], zorder=10-i, lw=0)

    # lgnd = plt.legend(loc='upper left')
    # for handle in lgnd.legendHandles[1:]:
    #     handle.set_sizes([200])

    # ax = plt.gca()
    # def xformat(x, pos=None): return lon2str(x)
    # def yformat(x, pos=None): return lat2str(x)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

    plt.show()
    plt.close()


    ####################################################
    ################ BARCHART TIMELINE #################
    ####################################################
    # targets = ('CH4_nM', 'CO2_uatm', 'Salinity', 'Temperature')
    # legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
    #                  'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
    #                  'Salinity':'Salinity, PSS',
    #                  'Temperature':'Temperature, C'}
    # date_labels = {3:'July 01, 2018', 4:'July 02, 2018', 0:'June 28, 2018', 1:'June 29, 2018', 2:'June 30, 2018'}
    # dates = ['28 June', '29 June', '30 June', '01 July', '02 July']
    # depth_inc = 0.25

    # avgs = []
    # stds = []
    # num_depths = int(np.ceil(5./depth_inc))
    # width = 0.04
    # depths = [depth_inc+i*depth_inc for i in range(0, num_depths)]
    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(depths))), axis=0)
    # last_depth = 0
    # ind = np.arange(len(dates))


    # for target in targets:
    #     last_depth = 0
    #     top_trend = []
    #     mid_trend = []
    #     bottom_trend = []
    #     for m in jy.mission[0:5]:
    #         top_trend.append(np.mean(m[(m['Depth'] <= 1.0) & (m['Depth'] > 0.0)][target]))
    #         mid_trend.append(np.mean(m[(m['Depth'] <= 2.5) & (m['Depth'] > 1.5)][target]))
    #         bottom_trend.append(np.mean(m[(m['Depth'] <= 5.0) & (m['Depth'] > 4.0)][target]))

    #     plt.figure(figsize=(15,3))
    #     for i, d in enumerate(depths):
    #         for j, m in enumerate(jy.mission[0:5]):
    #             tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
    #             avgs.append(np.mean(tmp[target]))
    #             stds.append(np.std(tmp[target]))
    #         # plt.plot(dates, avgs, c=colors[i], label=str(last_depth) + 'm-' + str(d) + 'm')
    #         # plt.errorbar(dates, avgs, yerr=stds, c=colors[i], label=str(last_depth) + 'm-' + str(d) + 'm')
    #         # plt.fill_between(dates, [a-s for a,s in zip(avgs,stds)], [a+s for a,s in zip(avgs, stds)], alpha=0.1, color=colors[i])
    #         plt.bar(ind + (width*i - (num_depths*width/2)),
    #                 avgs,
    #                 yerr=stds,
    #                 color=colors[i],
    #                 width=width,
    #                 label=str(last_depth) + '-' + str(d) + 'm',
    #                 error_kw={'ecolor':'red', 'elinewidth':0.5})
    #         last_depth = d
    #         avgs = []
    #         stds = []
    #     plt.plot(ind + (width*1-(num_depths*width/2)), top_trend, c=colors[1], marker='o', lw=3, linestyle='--', mec='k', ms=10)
    #     plt.plot(ind + (width*8-(num_depths*width/2)), mid_trend, c=colors[8], marker='o', lw=3, linestyle='--', mec='k', ms=10)
    #     plt.plot(ind + (width*18-(num_depths*width/2)), bottom_trend, c=colors[18], marker='o', lw=3, linestyle='--', mec='k', ms=10)

    #     plt.xlabel('Date', fontsize=16)
    #     plt.gca().set_xticks(ind)
    #     plt.gca().set_xticklabels(dates, fontsize=14)
    #     plt.ylabel(legend_labels[target])
    #     box = plt.gca().get_position()
    #     plt.gca().set_position([box.x0, box.y0, box.width*0.8, box.height])
    #     # plt.gca().set_aspect((rmax-rmin+100.)/(5.1))
    #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    #     plt.show()
    #     plt.close()



    ####################################################
    ################ GENERAL CASCADES ##################
    ####################################################
    # targets = ('CH4_nM', 'CO2_uatm', 'Salinity', 'Temperature')
    # legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
    #                  'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
    #                  'Salinity':'Salinity, PSS',
    #                  'Temperature':'Temperature, C'}
    # date_labels = {3:'July 01, 2018', 4:'July 02, 2018', 0:'June 28, 2018', 1:'June 29, 2018', 2:'June 30, 2018'}
    # dates = ['28 June', '29 June', '30 June', '01 July', '02 July']
    # depth_inc = 0.25
    
    # avgs = []
    # stds = []
    # ds = []
    # depths = [depth_inc+i*depth_inc for i in range(0,int(np.ceil(6/depth_inc)))]
    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(dates))), axis=0)
    # last_depth = 0

    # for target in targets:
    #     last_depth = 0
    #     for i, m in enumerate(jy.mission[0:5]):
    #         for d in depths:
    #             tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
    #             avgs.append(np.mean(tmp[target]))
    #             stds.append(np.std(tmp[target]))
    #             ds.append(np.mean(tmp['Depth']))
    #             last_depth = d
    #         plt.errorbar(avgs, ds, xerr=stds, c=colors[i], label=dates[i], marker='o', elinewidth=1.2, markersize=15, mec='k', lw=0)
    #         avgs = []
    #         stds = []
    #         ds = []
    #     plt.ylabel('Depth, m', fontsize=32)
    #     plt.gca().invert_yaxis()
    #     plt.xlabel(legend_labels[target], fontsize=32)
    #     plt.legend(loc='best', fontsize=28)
    #     plt.xticks(fontsize=28)
    #     plt.show()
    #     plt.close()

    ####################################################
    ################ REGIONAL CASCADES #################
    ####################################################
    # targets = ('CH4_nM', 'CO2_uatm', 'Salinity', 'Temperature')
    # legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
    #                  'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
    #                  'Salinity':'Salinity, PSS',
    #                  'Temperature':'Temperature, C'}
    # date_labels = {3:'July 01, 2018', 4:'July 02, 2018', 0:'June 28, 2018', 1:'June 29, 2018', 2:'June 30, 2018'}
    # dates = ['28 June', '29 June', '30 June', '01 July', '02 July']
    # depth_inc = 0.25
    # avgs = []
    # stds = []
    # ds = []
    # depths = [depth_inc+i*depth_inc for i in range(0,int(np.ceil(6/depth_inc)))]
    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(dates))), axis=0)
    # last_depth = 0

    # #get plotting values
    # vmin = []
    # vmax = []
    # for target in targets:
    #     temp_min = []
    #     temp_max = []
    #     for m in jy.mission:
    #         temp_min.append(np.nanmin(m[target].values))
    #         temp_max.append(np.nanmax(m[target].values))
    #     vmin.append(np.nanmin(temp_min))
    #     vmax.append(np.nanmax(temp_max))

    # #set the regions of interest, based on distance from freshwater creek mouth
    # regions = [3000, 2000, 1000, 500, 0, -1000]
    # n_regions = len(regions)-1
    # last_r = regions[0]

    # for k, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, n_regions, sharey=True, sharex=True, figsize=(20,7))
    #     for j, region in enumerate(regions[1:]):
    #         for i, m in enumerate(jy.mission[0:5]):
    #             for d in depths:
    #                 tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
    #                 tmp = tmp[(tmp['Distance'] <= last_r) & (tmp['Distance'] > region)]
    #                 avgs.append(np.nanmean(tmp[target]))
    #                 stds.append(np.nanstd(tmp[target]))
    #                 ds.append(np.nanmean(tmp['Depth']))
    #                 last_depth = d
    #             # ax[j].scatter(avgs, ds, c=colors[i], label=dates[i])
    #             if j == 0:
    #                 ax[j].errorbar(avgs, ds, xerr=stds,
    #                                c=colors[i], label=dates[i],
    #                                marker='o', lw=0,
    #                                elinewidth=1, markersize=10,
    #                                mec='k')
    #             else:
    #                 ax[j].errorbar(avgs, ds, xerr=stds,
    #                                c=colors[i], label='',
    #                                marker='o', lw=0,
    #                                elinewidth=1, markersize=10,
    #                                mec='k')
    #             first = np.nanmin([last_r, region])
    #             last = np.nanmax([last_r, region])
    #             if first == -1000:
    #                 ax[j].set_title('Embayment')
    #             else:
    #                 ax[j].set_title(str(first)+'-'+str(last)+'m')
    #             ax[j].set_ylim(0, 6)
    #             ax[j].set_xlim(vmin[k]-vmax[k]*0.05, vmax[k]+vmax[k]*0.05)
    #             ax[j].set_aspect((vmax[k]-vmin[k]+2*vmax[k]*0.05)/(5.1))
    #             avgs = []
    #             ds = []
    #             stds = []
    #         last_r = region
    #     last_r = regions[0]
    #     fig.text(0.5, 0.16, legend_labels[target], va='center', ha='center')
    #     fig.text(0.02, 0.5, 'Depth, m', va='center', ha='center', rotation='vertical')
    #     # fig.text(0.9, 0.37, 'Embayment', va='center', ha='center', color='grey', fontsize=14)
    #     plt.gca().invert_yaxis()
    #     lgnd = fig.legend(loc='lower center', ncol=len(dates))
    #     for handle in lgnd.legendHandles:
    #         print handle
    #         try:
    #             handle.numpoints = 100
    #         except:
    #             pass
    #     fig.subplots_adjust(bottom=0.04, top=0.98, left=0.05, right=0.95)

    #     plt.show()
    #     plt.close()

    # Examine the ST plots
    # for i, m in enumerate([jy.mission[2]]):
    #     jviz.st_plots(m['Salinity'].values, m['Temperature'].values,
    #                   m['CH4_umolkg'].values, 'CH4 (umol kg)', '')

    ###### DISTANCE BARCHART #######
    # targets = ('CH4_nM', 'CO2_uatm', 'Salinity', 'Temperature')
    # legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
    #                  'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
    #                  'Salinity':'Salinity, PSS',
    #                  'Temperature':'Temperature, C'}
    # transect_labels = {3:'July 01, 2018', 4:'July 02, 2018', 0:'June 28, 2018', 1:'June 29, 2018', 2:'June 30, 2018'}
    # # transects = ['Transect A', 'Transect B', 'Transect C', 'Transect D', 'Transect E']
    # distance_inc = 100.

    # avgs = []
    # stds = []
    # num_dists = int(np.ceil((2600)/distance_inc))
    # width = 0.03
    # dists = [0+i*distance_inc for i in range(0, num_dists)]
    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,len(dists))), axis=0)
    # last_dist = 0
    # ind = np.arange(len(jy.mission[:-1]))


    # for target in targets:
    #     last_dist = 0
    #     top_trend = []
    #     mid_trend = []
    #     bottom_trend = []
    #     for m in jy.mission[:-1]:
    #         top_trend.append(np.mean(m[(m['Distance'] <= 100.0) & (m['Distance'] > 0.0) & (m['Depth'] < 1.5)][target]))
    #         mid_trend.append(np.mean(m[(m['Distance'] <= 1050.) & (m['Distance'] > 950.) & (m['Depth'] < 1.5)][target]))
    #         bottom_trend.append(np.mean(m[(m['Distance'] <= 2050.) & (m['Distance'] > 1950.) & (m['Depth'] < 1.5)][target]))

    #     plt.figure(figsize=(15,3))
    #     for i, d in enumerate(dists[1:]):
    #         for j, m in enumerate(jy.mission[:-1]):
    #             tmp = m[(m['Distance'] <= d) & (m['Distance'] > last_dist) & (m['Depth'] < 1.5)]
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
    #     plt.plot(ind + (width*1-(num_dists*width/2)), top_trend, c=colors[1], marker='o', lw=3, linestyle='--', mec='k', ms=10)
    #     plt.plot(ind + (width*10-(num_dists*width/2)), mid_trend, c=colors[10], marker='o', lw=3, linestyle='--', mec='k', ms=10)
    #     plt.plot(ind + (width*20-(num_dists*width/2)), bottom_trend, c=colors[20], marker='o', lw=3, linestyle='--', mec='k', ms=10)

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
    ################ PROFILES ##########################
    ####################################################
    # ''' Draws the path of the vehicle each day as distance from the boat launch '''
    # targets = ('CH4_nM', 'CO2_uatm', 'Salinity', 'Temperature')
    # legend_labels = {'CH4_nM':'CH$_4$ Concentration, nM',
    #                  'CO2_uatm':'CO$_2$ Concentration, $\mu$atm',
    #                  'Salinity':'Salinity, PSS',
    #                  'Temperature':'Temperature, C'}
    # date_labels = {3:'July 01, 2018', 4:'July 02, 2018', 0:'June 28, 2018', 1:'June 29, 2018', 2:'June 30, 2018'}
    # dock_reference = (69.121595, -105.019215)
    # dock_reference = (69.125566, -105.004188)

    # # get the plotting settings for the values
    # vmin = []
    # vmax = []
    # rmin = 10000
    # rmax = -10000
    # for target in targets:
    #     temp_min = []
    #     temp_max = []
    #     for m in jy.mission:
    #         temp_min.append(np.nanmin(m[target].values))
    #         temp_max.append(np.nanmax(m[target].values))
    #     vmin.append(np.nanmin(temp_min))
    #     vmax.append(np.nanmax(temp_max))

    # for m in jy.mission:
    #     rmin = np.nanmin([rmin, np.nanmin(m['Distance'].values)])
    #     rmax = np.nanmax([rmax, np.nanmax(m['Distance'].values)])

    # # plt.plot(jy.mission[4]['Depth'])
    # # plt.show()
    # # plt.close()
    # # aa

    # # generate plots
    # date_labels = ['29 June', '2 July']
    # indices = [(1700, 3635), (3410, 5200)]
    # for i, target in enumerate(targets):
    #     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 8))
    #     for j, m in enumerate([jy.mission[1]]):#, jy.mission[4]]):
    #         # m = m.sort_values(by=target, ascending=True)
    #         m = m.iloc[indices[j][0]:indices[j][1]]
    #         scat = ax[j].scatter(m[target], m['Depth'], rasterized=False, s=10)

    #         # scat = ax[j].scatter(m['Distance'], m['Depth'], c=m[target], cmap='viridis', vmin=vmin[i], vmax=vmax[i], s=1, rasterized=True)
    #         # ax[j].axis([rmin-50.0, rmax+50.0, -0.1, 6])
    #         # ax[j].axvline(0, 0, 10, c='r', linestyle='--')
    #         # ax[j].annotate('Towards Receding Ice Edge', xy=(rmin, 9.8), ha='left', va='bottom', color='grey', rotation=90.)
    #         # ax[j].annotate('Towards Freshwater Creek', xy=(rmax, 9.8), ha='right', va='bottom', color='grey', rotation=270.)
    #         ax[j].set_title(date_labels[j], fontsize=36)
    #         ax[j].set_xlabel(target)
    #         # ax[j].set_aspect((rmax-rmin+100.)/(6.1))
    #     fig.text(0.03, 0.5, 'Depth, m', va='center', ha='center', rotation='vertical')
    #     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.07, right=0.87, wspace=0.1)
    #     plt.gca().invert_yaxis()
    #     plt.show()
    #     plt.close()
