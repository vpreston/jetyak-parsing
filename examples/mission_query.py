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


def process_wind(filename):
    ''' Helper method to process wind data from Cambridge Bay deployment '''
    df = pd.read_table(filename, delimiter=',', header=0, engine='c')
    df.loc[:, 'Minute'] = 0.0
    df.loc[:, 'Hour'] = df['Time'].str.split(':').str.get(0).astype('float')+6.0
    df.loc[:, 'Second'] = 0.0
    df = sensors.make_global_time(df)
    return df

if __name__ == '__main__':

    ######## Make a "processing" JetYak ################
    # Data to access
    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.28.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # gga_dirs = [base_path + 'gga/2018-06-28/gga_2018-06-28_f0001.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330013612.txt']
    # mission_name = '0628.csv'
    # trim_values = [[2458298.3855, 2458298.3845], [2458298.394, 2458298.3925], [2458298.422, 2458298.42]]
    # bounds = [2458298.339353009, 2458298.4214641205]

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.29.2018/data/'
    # ctd_dirs = [base_path+'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path+'airmar/airmar_20180330034652.txt', base_path+'airmar/airmar_20180330082958.txt']
    # gga_dirs = [base_path+'gga/2018-06-29/gga_2018-06-29_f0002.txt']
    # mission_name = '0629.csv'
    # trim_values = [[2458299.3626, 2458299.362]]
    # bounds = [2458299.200787037, 2458299.410834491]

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/06.30.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330093911.txt']
    # gga_dirs = [base_path + 'gga/2018-06-30/gga_2018-06-30_f0001.txt']
    # mission_name = '0630.csv'
    # trim_values = [[2458300.382, 2458300.3813]]
    # bounds = [2458300.3061458333, 2458300.456943287]

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.01.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330132521.txt']
    # gga_dirs = [base_path + 'gga/2018-07-01/gga_2018-07-01_f0001.txt']
    # mission_name = '0701.csv'
    # trim_values = None
    # bounds = [2458301.2410046295, 2458301.4014780093]

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.02.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330172728.txt']
    # gga_dirs = [base_path + 'gga/2018-07-02/gga_2018-07-02_f0001.txt', base_path + 'gga/2018-07-02/gga_2018-07-02_f0002.txt']
    # mission_name = '0702.csv'
    # trim_values = [[2458302.186, 2458302.184]]
    # bounds = [2458302.171130787, 2458302.3713055556]

    # base_path = '/home/vpreston/Documents/field_work/cb_2018/Cambridge-Bay-06.2018/07.04.2018/data/'
    # ctd_dirs = [base_path + 'ctd/rbr_data/rbr_data_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180330223042.txt']
    # gga_dirs = [base_path + 'gga/2018-07-04/gga_2018-07-04_f0003.txt', base_path + 'gga/2018-07-04/gga_2018-07-04_f0004.txt']
    # mission_name = '0704.csv'
    # trim_values = [[2458304.39, 2458304.352],[2458304.3085, 2458304.3077],[2458304.256, 2458304.2547],
    #               [2458304.304, 2458304.303],[2458304.3095,2458304.308], [2458304.347, 2458304.345]]
    # bounds = [2458304.2128425925, 2458304.345]

    # Make a JetYak
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds)
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # # print np.sort(jy.airmar.get_df()['Julian_Date'].values)[0], np.sort(jy.airmar.get_df()['Julian_Date'].values)[-1]

    # # Can now perform work with the sensors
    # jy.create_mission({'geoframe':'airmar'})
    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/', mission_name=mission_name)

    ###### Make a mission "analyzing" JetYak ###########
    # Data to access
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/'
    miss = ['mission_0628.csv', 'mission_0629.csv', 'mission_0630.csv',
            'mission_0701.csv', 'mission_0702.csv', 'mission_0704.csv']
    titles = ['June 28', 'June 29', 'June 30', 'July 1', 'July 2', 'July 4']
    # miss = ['mission_0628.csv', 'mission_0629.csv']

    # Create mission operator
    jy = jetyak.JetYak()
    # jy.load_mission([base_path+m for m in miss], header=[0,1], simplify_mission=True)

    # Create windspeed queries
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



    # jy.save_mission(base_path, 'trimmed')
    jy.load_mission([base_path+'trimmed_'+str(i)+'.csv' for i in [0,1,2,3,4,5]], header=0, simplify_mission=False)
    jy.add_bottle_samples('/home/vpreston/Documents/IPP/cb-methane-data/discrete_samples.csv')


    # Examine bottle samples and JetYak data
    jviz.compare_samples(jy, geo_epsilon=50., depth_epsilon=0.25, save_path=base_path)

    # Examine the ST plots
    # for i, m in enumerate(jy.mission):
    #     jviz.st_plots(m['Salinity'].values, m['Temperature'].values,
    #                   m['CH4_ppm'].values, 'CH4 (ppm)', titles[i])

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
    # for m in jy.mission:
    #     if np.nanmax(m['CH4_ppm'].values) > most_meth:
    #         most_meth = np.nanmax(m['CH4_ppm'].values)
    #     if np.nanmin(m['CH4_ppm'].values) < least_meth:
    #         least_meth = np.nanmin(m['CH4_ppm'].values)

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

    # # # Create filled contours
    # for m in [jy.mission[1], jy.mission[-2]]:
    #     print m.head(5)
    #     jviz.filled_contours(m, region=r, buff=0.003, target='CH4_ppm', depth_lim=1.0, vmin=least_meth, vmax=most_meth)