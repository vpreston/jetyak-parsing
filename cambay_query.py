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
        return np.sqrt(dist)
    except:
        return None


if __name__ == '__main__':
    ######## Make a "processing" JetYak ################
    # Data to access
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.28.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # gga_dirs = [base_path + 'gga/2018-06-28/gga_2018-06-28_f0001.txt']
    # op_dirs = [base_path + 'op/optode_20180330013534.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330013612.txt']
    # mission_name = '0628.csv'
    # trim_values = [[2458298.3855, 2458298.3845], [2458298.394, 2458298.3925], [2458298.422, 2458298.42]]
    # bounds = [2458298.339353009, 2458298.4214641205]
    # offset = 2440678.4842

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.29.2018/data/'
    # ctd_dirs = [base_path+'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path+'airmar/airmar_20180330034652.txt', base_path+'airmar/airmar_20180330082958.txt']
    # gga_dirs = [base_path+'gga/2018-06-29/gga_2018-06-29_f0002.txt']
    # op_dirs = [base_path + 'op/optode_20180330034739.txt', base_path + 'op/optode_20180330082905.txt']
    # mission_name = '0629.csv'
    # trim_values = [[2458299.3626, 2458299.362]]
    # bounds = [2458299.200787037, 2458299.410834491]
    # offset = 2440679.255

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.30.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330093911.txt']
    # gga_dirs = [base_path + 'gga/2018-06-30/gga_2018-06-30_f0001.txt']
    # op_dirs = [base_path + 'op/optode_20180330093807.txt']
    # mission_name = '0630.csv'
    # trim_values = [[2458300.382, 2458300.3813]]
    # bounds = [2458300.3061458333, 2458300.456943287]
    # offset = 2440680.11575

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.01.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330132521.txt']
    # gga_dirs = [base_path + 'gga/2018-07-01/gga_2018-07-01_f0001.txt']
    # op_dirs = [base_path + 'op/optode_20180330132615.txt']
    # mission_name = '0701.csv'
    # trim_values = None
    # bounds = [2458301.2410046295, 2458301.4014780093]
    # offset = 2440680.893

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.02.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330172728.txt']
    # gga_dirs = [base_path + 'gga/2018-07-02/gga_2018-07-02_f0001.txt', base_path + 'gga/2018-07-02/gga_2018-07-02_f0002.txt']
    # op_dirs = [base_path + 'op/optode_20180330172646.txt']
    # mission_name = '0702.csv'
    # trim_values = [[2458302.186, 2458302.184]]
    # bounds = [2458302.171130787, 2458302.3713055556]
    # offset = 2440681.650

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.04.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330223042.txt']
    # gga_dirs = [base_path + 'gga/2018-07-04/gga_2018-07-04_f0003.txt', base_path + 'gga/2018-07-04/gga_2018-07-04_f0004.txt']
    # op_dirs = [base_path + 'op/optode_20180330223020.txt']
    # mission_name = '0704.csv'
    # trim_values = [[2458304.39, 2458304.352],[2458304.3085, 2458304.3077],[2458304.256, 2458304.2547],
    #               [2458304.304, 2458304.303],[2458304.3095,2458304.308], [2458304.347, 2458304.345]]
    # bounds = [2458304.2128425925, 2458304.345]
    # offset = 2440683.486




    # #### Make a JetYak
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset])
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # jy.attach_sensor('optode', op_dirs)
    # # print np.sort(jy.airmar.get_df()['Julian_Date'].values)[0], np.sort(jy.airmar.get_df()['Julian_Date'].values)[-1]

    # # Can now perform work with the sensors
    # jy.create_mission({'geoframe':'airmar'})
    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/cambay/', mission_name=mission_name)





    # ###### Make a mission "analyzing" JetYak ###########
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/cambay/'
    miss = ['0628.csv', '0629.csv', '0630.csv',
            '0701.csv', '0702.csv', '0704.csv']
    titles = ['June 28', 'June 29', 'June 30', 'July 1', 'July 2', 'July 4']

    # # Create mission operator
    jy = jetyak.JetYak()

    #### RUN TO PARSE NEW MISSIONS
    jy.load_mission([base_path+m for m in miss], header=[0,1], simplify_mission=True, meth_eff=0.091, carb_eff=0.70)
    jy.save_mission(base_path, 'trimmed_ch4_9_1')


    #### CREATE WINDSPEED QUERIES (FROM UNSIMPLIFIED MISSIONS)
    # filepath = '/home/vpreston/Documents/IPP/jetyak_parsing/airport_wind.csv'
    # wind = process_wind(filepath)

    # for i in range(0,len(miss)):
    #     print jy.mission[i]['airmar'].columns
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

    #### RUN TO PARSE SIMPLIFIED SAVED MISSIONS AND BOTTLE SAMPLES
    jy.load_mission([base_path+'trimmed_ch4_9_1_'+str(i)+'.csv' for i in [0,1,2,3,4,5]], header=0, simplify_mission=False)
    jy.add_bottle_samples('/home/vpreston/Documents/IPP/cb-methane-data/discrete_samples.csv')
    

    #### LOOK AT JULY1 CTD CAST and GENERAL CAST DATA
    # cast = jy.mission[3]
    # cast = cast[(np.fabs(cast['Latitude']-69.105) < 0.01) & (np.fabs(cast['Longitude']--105.0417) < 0.01)]
    # np.savetxt('temp.csv', cast['Depth'].values)
    # np.savetxt('tempsal.csv', cast['Salinity'].values)

    # for m in jy.mission:
    #     m = m[(m['Depth'] <= 0.7) & (m['Depth'] >= 0.5)]
    #     # m = m[(m['Depth'] <= 3.1) & (m['Depth'] >= 2.9)]
    #     print np.mean(m['Temperature'].values)


    #### LOOK AT BOTTLE SAMPLES
    collapsed = jy.collapse_bottle_samples()
    matched = jy.match_bottles(collapsed, geo_epsilon=50., depth_epsilon=0.25)
    print matched.head(5)
    print jy.mission[0].head(5)

    matched = matched.dropna()
    groups = matched.groupby('day')
    for name, group in groups:
        # plt.errorbar(group['ch4_mean'], group['jch4_mean'], xerr=group['ch4_std'], yerr=group['jch4_std'], fmt='o', label=name)
        plt.errorbar(group['co2_mean'], group['jco2_mean'], xerr=group['co2_std'], yerr=group['jco2_std'], fmt='o', label=name)
    # plt.plot(np.linspace(0,400,10), np.linspace(0,400,10), 'k--', label='One-to-One')
    plt.plot(np.linspace(0,2000,10), np.linspace(0,2000,10), 'k--', label='One-to-One')

    # mask = ~np.isnan(matched['ch4_mean']) & ~np.isnan(matched['jch4_mean'])
    mask = ~np.isnan(matched['co2_mean']) & ~np.isnan(matched['jco2_mean'])

    # slope, intercept, r_value, p_value, std_err = stats.linregress(matched['ch4_mean'].values[mask],matched['jch4_mean'].values[mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(matched['co2_mean'].values[mask],matched['jco2_mean'].values[mask])

    print slope, r_value
    
    # line = slope*np.linspace(0,400,10)+intercept
    # plt.plot(np.linspace(0,400,10), line, 'r--', label='Best Line of Fit')
    line = slope*np.linspace(0,2000,10)+intercept
    plt.plot(np.linspace(0,2000,10), line, 'r--', label='Best Line of Fit')

    # x = matched['ch4_mean'].values[:,np.newaxis]
    # a, _, _, _ = np.linalg.lstsq(x,matched['jch4_mean'].values[mask])
    x = matched['co2_mean'].values[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(x,matched['jco2_mean'].values[mask])
    print a
    # plt.plot(np.linspace(0,400,10), np.linspace(0,400,10)*a, 'g--', label='Best Line of Fit, Forced Origin')
    plt.plot(np.linspace(0,2000,10), np.linspace(0,2000,10)*a, 'g--', label='Best Line of Fit, Forced Origin')


    plt.axis('square')
    plt.legend()
    # plt.xlabel('CH4 Bottle Sample, nM')
    # plt.ylabel('CH4 ChemYak Observation, nM')
    plt.xlabel('CO2 Bottle Sample, uatm')
    plt.ylabel('CO2 ChemYak Observation, utam')

    plt.show()
    plt.close()

    # # Create timeseries
    # target = 'CH4_umolkg' #'Temperature' #'Salinity' #'CO2_uatm' #'CH4_umolkg'
    # target_constant = 1e3 #1
    # avgs = []
    # stds = []
    # depths = [0.25+i*0.25 for i in range(0,24)]
    # colors = plt.cm.viridis(np.linspace(0,1,len(depths)))
    # dates = ['June 28', 'June 29', 'June 30', 'July 1', 'July 2']
    # last_depth = 0

    # for i, d in enumerate(depths):
    #     for m in jy.mission:
    #         tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
    #         avgs.append(np.mean(tmp[target]*target_constant))
    #         stds.append(np.std(tmp[target]*target_constant))
    #     plt.plot(dates, avgs, c=colors[i], label=str(last_depth) + 'm-' + str(d) + 'm')
    #     plt.fill_between(dates, [a-s for a,s in zip(avgs,stds)], [a+s for a,s in zip(avgs, stds)], alpha=0.1, color=colors[i])
    #     last_depth = d
    #     avgs = []
    #     stds = []

    # plt.xlabel('Date')
    # plt.ylabel('Methane, nmol kg$^{-1}$')
    # # plt.ylabel('pCO2, $\mu$atm')
    # # plt.ylabel('Salinity, PSS')
    # # plt.ylabel('Temperature, C')
    # plt.legend()
    # plt.show()
    # plt.close()


    # #Create cascades
    # target = 'Salinity' #'Temperature' #'Salinity' #'CO2_uatm' #'CH4_umolkg'
    # avgs = []
    # ds = []
    # depths = [0.25+i*0.25 for i in range(0,24)]
    # dates = ['June 28', 'June 29', 'June 30', 'July 1', 'July 2']
    # colors = plt.cm.viridis(np.linspace(0,1,len(dates)))
    # last_depth = 0

    # for i, m in enumerate(jy.mission):
    #     print i
    #     for d in depths:
    #         tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
    #         avgs.append(np.mean(tmp[target]))
    #         ds.append(np.mean(tmp['Depth']))
    #         last_depth = d
    #     plt.scatter(avgs, ds, c=colors[i], label=dates[i])
    #     avgs = []
    #     ds = []

    # plt.ylabel('Depth, m')
    # plt.gca().invert_yaxis()
    # # plt.xlabel('Methane, nmol kg$^{-1}$')
    # # plt.xlabel('pCO2, $\mu$atm')
    # plt.xlabel('Salinity, PSS')
    # # plt.xlabel('Temperature, C')
    # plt.legend()
    # plt.show()
    # plt.close()

    # #Create regional cascades (divide by x-axis)
    # n_regions = 5
    # target = 'Temperature'
    # target_constant = 1#1e3 #1
    # depths = [0.10+i*0.10 for i in range(0,60)]
    # dates = ['June 28', 'June 29', 'June 30', 'July 1', 'July 2']
    # colors = plt.cm.viridis(np.linspace(0,1,len(dates)))
    # avgs = []
    # ds = []
    # last_depth = 0
    # x_min = 1000
    # x_max = -1000
    # y_min = 1000
    # y_max = -1000
    # m_min =  1000
    # m_max = -1000
    # c_min = 1000
    # c_max = -1000
    # for m in jy.mission:
    #     # print np.nanmin(m['Longitude'])
    #     # print np.nanmax(m['Longitude'])
    #     x_min = min(x_min, np.nanmin(m['Longitude']))
    #     y_max = max(y_max, np.nanmax(m['Latitude']))

    #     m_min = min(m_min, np.nanmin(m['CH4_umolkg']))
    #     m_max = max(m_max, np.nanmax(m['CH4_umolkg']))
    #     c_min = min(c_min, np.nanmin(m['CO2_uatm']))
    #     c_max = max(c_max, np.nanmax(m['CO2_uatm']))

    #     if np.nanmax(m['Longitude']) >= 0.0:
    #         pass
    #     else:
    #         x_max = max(x_max, np.nanmax(m['Longitude']))
    #     if np.nanmin(m['Latitude']) <= 0.0:
    #         pass
    #     else:
    #         y_min = min(y_min, np.nanmin(m['Latitude']))

    # print y_min, y_max

    # r = np.linspace(x_min, x_max, n_regions+1)
    # r.sort()
    # last_r = x_min

    # base = Basemap(llcrnrlon=x_min-0.01, llcrnrlat=y_min-0.01, urcrnrlon=x_max+0.01, urcrnrlat=y_max+0.01,
    #                resolution='l', projection='cyl')

    # base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    # proj_lon, proj_lat = base(*(r, [(y_min+y_max)/2 for q in r]))

    # scat = base.scatter(proj_lon, proj_lat)
    # # plt.show()
    # plt.close()


    # fig, ax = plt.subplots(1, n_regions, sharey=True, sharex=True)

    # for j, region in enumerate(r[1:]):
    #     for i, m in enumerate(jy.mission):
    #         for d in depths:
    #             tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
    #             tmp = tmp[(tmp['Longitude'] > last_r) & (tmp['Longitude'] <= region)]
    #             avgs.append(np.mean(tmp[target]*target_constant))
    #             ds.append(np.mean(tmp['Depth']))
    #             last_depth = d
    #         ax[j].scatter(avgs, ds, c=colors[i], label=dates[i])
    #         avgs = []
    #         ds = []
    #     last_r = region

    #     ax[j].set_ylabel('Depth, m')
    #     plt.gca().invert_yaxis()
    #     # ax[j].set_xlabel('Methane, nmol kg$^{-1}$')
    #     # ax[j].set_xlabel('pCO2, $\mu$atm')
    #     # ax[j].set_xlabel('Salinity, PSS')
    #     ax[j].set_xlabel('Temperature, C')
    # plt.legend()
    # plt.show()
    # plt.close()


    #create daily tracks
    # quants = ['Salinity', 'CH4_umolkg', 'CO2_uatm']
    # labels = ['Salinity, PSS', 'Methane, nmol kg$^{-1}$', 'pCO2, $\mu$atm']
    # for m in jy.mission:
    #     m = m[m['Latitude']!= 0.0]
    #     plt.figure()
    #     base = Basemap(llcrnrlon=x_min-0.001, llcrnrlat=y_min-0.001, urcrnrlon=x_max+0.001, urcrnrlat=y_max+0.001,
    #                resolution='l', projection='cyl')
    #     base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    #     m = m[m['Depth'] < 1.0]
    #     proj_lon, proj_lat = base(*(m['Longitude'], m['Latitude']))
    #     base.scatter(proj_lon, proj_lat, s=0.05, c='k', alpha=0.5)
    #     plt.show()
    #     plt.close()
    #     for i in range(0,len(quants)):
    #         scat = plt.scatter(m['Longitude'], m['Depth'], c=m[quants[i]]*1000, cmap='viridis', alpha=1.0)
    #         plt.xlabel('Longitude')
    #         plt.xlim(x_min-0.01, x_max+0.01)
    #         plt.ylabel('Depth, m')
    #         plt.gca().invert_yaxis()
    #         cbar = plt.colorbar(scat)
    #         cbar.set_label(labels[i])
    #         plt.show()
    #         plt.close()

    # #create top meter tracks
    # m_min =  1000
    # m_max = -1000
    # c_min = 1000
    # c_max = -1000
    # m = m[m['Depth'] <= 1.0]
    # for m in jy.mission:
    #     m_min = min(m_min, np.nanmin(m['CH4_umolkg']))
    #     m_max = max(m_max, np.nanmax(m['CH4_umolkg']))
    #     c_min = min(c_min, np.nanmin(m['CO2_uatm']))
    #     c_max = max(c_max, np.nanmax(m['CO2_uatm']))

    # for m in jy.mission:
    #     m = m[m['Latitude']!= 0.0]
    #     plt.figure()
    #     base = Basemap(llcrnrlon=x_min-0.001, llcrnrlat=y_min-0.001, urcrnrlon=x_max+0.001, urcrnrlat=y_max+0.001,
    #                resolution='l', projection='cyl')
    #     base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    #     proj_lon, proj_lat = base(*(m['Longitude'], m['Latitude']))
    #     base.scatter(proj_lon, proj_lat, s=0.05, c=m['CH4_umolkg'], cmap='viridis', vmin=m_min, vmax=m_max)
    #     plt.show()
    #     plt.close()


    # Examine the ST plots
    # for i, m in enumerate([jy.mission[2]]):
    #     jviz.st_plots(m['Salinity'].values, m['Temperature'].values,
    #                   m['CH4_umolkg'].values, 'CH4 (umol kg)', '')


    #######################
    # Create cascades
    # jviz.val_depth_cascades(jy.mission, depth_diff=0.1, limit=5.0)
    # regions = shapefile.Reader('./regions.shp')
    # jviz.regional_comparison(jy.mission, regions, depth_diff=0.1, limit=5.0)

    # most_meth = 0
    # least_meth = 100
    # xmin = 1000
    # xmax = -1000
    # ymin = 1000
    # ymax = -1000
    # target = 'Salinity'
    # for m in [jy.mission[2]]:
    #     if np.nanmax(m[target].values) > most_meth:
    #         most_meth = np.nanmax(m[target].values)
    #     if np.nanmin(m[target].values) < least_meth:
    #         least_meth = np.nanmin(m[target].values)

    # for m in [jy.mission[-2]]:
    #     if np.nanmax(m['Latitude']) > xmax:
    #         xmax = np.nanmax(m['Latitude'].values)
    #     if np.nanmin(m['Latitude']) < xmin:
    #         xmin = np.nanmin(m['Latitude'].values)
    #     if np.nanmax(m['Longitude']) > ymax:
    #         ymax = np.nanmax(m['Longitude'].values)
    #     if np.nanmin(m['Longitude']) < ymin:
    #         ymin = np.nanmin(m['Longitude'].values)

    # r = [xmin, xmax, ymin, ymax]
    # print r

    # lat, lon = 69.108784, -105.048985

    # # Create filled contours
    # for m in [jy.mission[2]]:
    #     print m.head(5)
    #     # jviz.filled_contours(m, region=r, buff=0.003, target=target, depth_lim=1.0, vmin=least_meth, vmax=most_meth)

    #     fig, ax = plt.subplots()
    #     ax.axis([500, 2750, -0.5, 6.5])
    #     ax.invert_yaxis()
    #     m.loc[:, 'Distance'] = m.apply(lambda x: get_distance((lat, lon), (x['Latitude'], x['Longitude'])), axis=1)
    #     # print m['Distance']
    #     scat = plt.scatter(m['Distance'], m['Depth'], c=m[target], s=10, cmap='viridis', lw=0, vmin=least_meth, vmax=most_meth)
    #     cbar = plt.colorbar(scat)
    #     cbar.set_label('Salinity', fontsize=20)
    #     plt.xlabel('Distance from Ice Edge (m)', fontsize=20)
    #     plt.ylabel('Depth (m)', fontsize=20)
    #     plt.show()
    #     plt.close()