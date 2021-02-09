#!/usr/env/python

'''
This is an example in which all JetYak data collected is compiled.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import numpy as np
import jetyak
import jviz
import shapefile
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ####################################################
    ###### Mission Data and Params #####################
    ####################################################
    
    ###### Cambridge Bay Data #######
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


    ###### Wareham Data #######
    # base_path = '/home/vpreston/Documents/field_work/wareham/rawdata/'
    # ctd_dirs = [base_path + 'ctd_data.txt']
    # airmar_dirs = [base_path + 'airmar1.txt', base_path + 'airmar2.txt', base_path + 'airmar3.txt']
    # gga_dirs = [base_path + 'gga_2017-07-12_f0000.txt', base_path + 'gga_2017-07-12_f0001.txt']
    # mission_name = 'WH.csv'
    # trim_values = None
    # bounds = None


    ###### New Bedford Data #######
    # base_path = '/home/vpreston/Documents/IPP/nb-effluent-plumes/data/'
    # ctd_dirs = [base_path + 'ctd/ctd_data.txt']
    # airmar_dirs = [base_path + 'airmar/airmar_20180329181245.txt', base_path + 'airmar/airmar_20180329191141.txt',
    #                base_path + 'airmar/airmar_20180329204336.txt', base_path + 'airmar/airmar_20180329213838.txt',
    #                base_path + 'airmar/airmar_20180329221731.txt', base_path + 'airmar/airmar_20180329230448.txt']
    # gga_dirs = [base_path + 'gga/gga_329_data.txt']
    # inner_mission_name = 'NB_inner.csv'
    # outer_mission_name = 'NB_outer.csv'
    # inner_bounds = [2458207+0.055, 2458207+0.205]
    # outer_bounds = [2458207+0.231, 2458207+0.274]
    # trim_values = None

    ####################################################
    ###### Make a "mission" JetYak #####################
    ####################################################
    # jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset])
    # jy.attach_sensor('ctd', ctd_dirs)
    # jy.attach_sensor('gga', gga_dirs)
    # jy.attach_sensor('airmar', airmar_dirs)
    # jy.attach_sensor('optode', optode_dirs)


    # # Can now perform work with the sensors
    # jy.create_mission({'geoframe':'airmar'})
    # jy.save_mission('/home/vpreston/Documents/IPP/jetyak_parsing/missions/falkor/', mission_name=mission_name)

    # print jy.mission[0].head(5)

    ####################################################
    ###### Make a mission "analyzing" JetYak ###########
    ####################################################
    # Data to access
    base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/falkor/'
    # miss = ['mission_0628.csv', 'mission_0629.csv', 'mission_0630.csv',
    #         'mission_0701.csv', 'mission_0702.csv', 'mission_0704.csv',
    #         'Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv',
    #         'NB_inner.csv', 'NB_outer.csv',
    #         'WH.csv']

    miss = ['Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv']

    # Create mission operator
    # jy = jetyak.JetYak()
    # jy.load_mission([base_path+m for m in miss], header=[0,1])
    # jy.save_mission(base_path, mission_name='trimmed')

    jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_0.csv', base_path+'trimmed_1.csv', base_path+'trimmed_2.csv'], header=0, simplify_mission=False)
    # jy.add_bottle_samples('/home/vpreston/Documents/field_work/new_bedford/discrete_samples.csv')

    smallest_seen = 10000.
    largest_seen = -10000.
    for m in jy.mission:
        if np.nanmin(m['CH4_ppm'].values) < smallest_seen:
            smallest_seen = np.nanmin(m['CH4_ppm'].values)
        if np.nanmax(m['CH4_ppm'].values) > largest_seen:
            largest_seen = np.nanmax(m['CH4_ppm'].values)

    target='O2'
    for m in jy.mission:
        # jviz.filled_contours(m, target=target, depth_lim=15.0, vmin=np.nanmin(m[target].values), vmax=np.nanmax(m[target].values))
        jviz.colored_scatter(m, target=target, depth_lim=15.0, vmin=32., vmax=np.nanmax(m[target].values))
        # jviz.st_plots(m['Salinity'].values, m['Temperature'].values, m['CH4_ppm'].values, target_label='CH4 ppm', title='ST Colored By Methane')

    # Examine CH4 and Temperature
    # missions = jy.mission
    # c = ['r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'g', 'g', 'm']
    # lab = ['Cambridge Bay', 'Falkor', 'New Bedford', 'Wareham']
    # last_c = 'q'
    # r=0
    # plt.figure()
    # for i, m in enumerate(missions):
    #     # print m.head(5)
    #     if last_c not in c[i]:
    #         plt.scatter(m['Temperature'], m['CH4_ppm'], c=c[i], linewidth=None, edgecolor='face', alpha=0.1, label=lab[r])
    #         last_c = c[i]
    #         r += 1
    #     else:
    #         plt.scatter(m['Temperature'], m['CH4_ppm'], c=c[i], linewidth=None, edgecolor='face', alpha=0.1, label='')
    # plt.xlabel('Temperature (C)')
    # plt.ylabel('Methane, Unmodified from GGA, ppm')
    # plt.title('All JetYak Deployments')
    # plt.legend()
    # plt.show()
    # plt.close()
    # jviz.compare_samples(jy, geo_epsilon=30.0, depth_epsilon=0.5, save_path=None)




