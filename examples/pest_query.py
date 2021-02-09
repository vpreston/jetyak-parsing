#!/usr/env/python

'''
This is an example in which all CTD and Optode data is compiled for PEST deployments.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import numpy as np
import jetyak
import sensors
import jviz
import shapefile
import matplotlib
import gpxpy
import pandas as pd
import copy
import gsw
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as mb
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.mlab import griddata
import utm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import LogFormatter


def lat2str(deg):
    min = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    dir = 'N'
    if deg < 0:
        if min != 0.0:
            deg += 1.0
            min -= 60.0
        dir = 'S'
    return ("%d$\degree$ %g'") % (np.abs(deg),np.abs(min))

def lon2str(deg):
    min = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    dir = 'E'
    if deg < 0:
        if min != 0.0:
            deg += 1.0
            min -= 60.0
        dir = 'W'
    return ("%d$\degree$ %g'") % (np.abs(deg),np.abs(min))

def calc_salinity(temp, conduc):
	pressure = 0.1
	salt = gsw.SP_from_C(conduc, temp, pressure)
	return salt



if __name__ == '__main__':

    ####################################################
    ###### Instrument Data and Params ##################
    ####################################################

    base_path = '/home/vpreston/Documents/field_work/pest/'
    save_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/pest/'
    ctd_dirs = [base_path + 'ctd_all.txt']
    optode_dirs = [base_path + 'op_all.txt']
    gps_dirs = [base_path + '20190605-142802.gpx']
    # ctd_dirs = [base_path + 'ctd.txt']
    # optode_dirs = [base_path + 'op_morning.txt']
    # gps_dirs = [base_path + '20190605-105758.gpx']
    mission_name = 'PEST_ExternalLoggers_060519_all.csv'
    trim_values = None
    bounds = None
    offset = 0.0

    ####################################################
    ###### Make a "mission" dataset ####################
    ####################################################

    jy = jetyak.JetYak(trim_vals=trim_values, bounds=bounds, args=[offset])
    jy.attach_sensor('ctd', ctd_dirs)
    jy.attach_sensor('mini_optode', optode_dirs)
    jy.attach_sensor('phone_gps', gps_dirs)

    # Can now perform work with the sensors
    jy.create_mission({'geoframe':'phone_gps'})
    jy.save_mission(save_path, mission_name=mission_name)

    # Also get the waypoints from the mission and save those
    data = []
    for point in jy.phone_gps.waypoints:
    	data.append([point.longitude, point.latitude, point.elevation, point.time, point.name])
    pdf = pd.DataFrame(data, columns=['Longitude', 'Latitude', 'Elevation', 'Time', 'Name'])

    # Set Times
    data = copy.copy(pdf)
    data.loc[:, 'Year'] = data.apply(lambda x: x['Time'].year, axis=1)
    data.loc[:, 'Month'] = data.apply(lambda x: x['Time'].month, axis=1)
    data.loc[:, 'Day'] = data.apply(lambda x: x['Time'].day, axis=1)
    data.loc[:, 'Hour'] = data.apply(lambda x: x['Time'].hour, axis=1)
    data.loc[:, 'Minute'] = data.apply(lambda x: x['Time'].minute, axis=1)
    data.loc[:, 'Second'] = data.apply(lambda x: x['Time'].second, axis=1)
    data = sensors.make_global_time(data)

    # Query for Data Reading
    m = jy.mission[0]
    oxy = []
    conduct = []
    salt = []
    ctdtemp = []
    optemp = []
    for lat, lon, time in zip(pdf.Latitude, pdf.Longitude, pdf.Julian_Date):
    	# temp = m[(m['phone_gps']['Latitude'] == lat) & (m['phone_gps']['Longitude'] == lon)]
    	temp = m[np.fabs(m.index - time) < 0.0007]
    	oxy.append(np.median(temp['mini_optode']['DO']))
    	conduct.append(np.median(temp['ctd']['Conductivity']))
    	salt.append(np.median(temp['ctd']['Salinity']))
    	ctdtemp.append(np.median(temp['ctd']['Temperature']))
    	optemp.append(np.median(temp['mini_optode']['Temperature']))

    data.loc[:, 'DO'] = oxy
    data.loc[:, 'Conductivity'] = conduct
    data.loc[:, 'Salinity'] = salt
    data.loc[:, 'CTD_Temperature'] = ctdtemp
    data.loc[:, 'Op_Temperature'] = optemp

    # print conduct
    # print data.Name

    pdf = data
    pdf.to_csv(save_path+'waypoint_'+mission_name)

    # plt.plot(m.index, m['ctd']['Temperature'], label='CTD')
    # plt.plot(m.index, m['mini_optode']['Temperature'], label='Optode')
    # plt.legend()
    # plt.xlabel('Julian Date')
    # plt.ylabel('Temeprature, C')
    # plt.show()
    # plt.close()

    x_min = np.nanmin(m['phone_gps']['Latitude']) - 0.0001
    x_max = np.nanmax(m['phone_gps']['Latitude']) + 0.0001
    y_min = np.nanmin(m['phone_gps']['Longitude']) - 0.0001
    y_max = np.nanmax(m['phone_gps']['Longitude']) + 0.0001

    # y_max = -70.640268
    # y_min = -70.641015
    # x_min = 41.575445
    # x_max = 41.576148


    base = Basemap(llcrnrlon=y_min,
                   llcrnrlat=x_min,
                   urcrnrlon=y_max,
                   urcrnrlat=x_max,
                   resolution='l',
                   projection='cyl',
                   suppress_ticks=False)

    # im = plt.imshow(plt.imread('/home/vpreston/Pictures/little_sipp.jpg'), extent=(y_min, y_max, x_min, x_max))

    proj_lon, proj_lat = base(*(m['phone_gps']['Longitude'].values, m['phone_gps']['Latitude'].values))
    path_lon, path_lat = base(*(pdf['Longitude'].values, pdf['Latitude'].values))

    scat = base.scatter(path_lon,
                        path_lat,
                        zorder=5,
                        s=100.0,
                        alpha=1.0,
                        c=pdf['Conductivity'],
                        cmap='coolwarm',
                        lw=0)
    base.scatter(proj_lon,
                 proj_lat,
                 zorder=3,
                 s=1.0,
                 alpha=1.0,
                 c='k')

    ax = plt.gca()
    def xformat(x, pos=None): return lon2str(x)
    def yformat(x, pos=None): return lat2str(x)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.gcf().tight_layout(rect=(0, 0.11, 1, 1))
    cbar = plt.colorbar(scat)
    # cbar.set_label('Salinity, PSS')
    cbar.set_label('Conductivity, mS/cm')
    # cbar.set_label('DO, mg/l')
    # cbar.set_label('Temperature, C')


    # plt.legend()
    plt.show()
    plt.close()

    scat = plt.scatter(pdf['DO'], pdf['Conductivity'], c=pdf['CTD_Temperature'], cmap='coolwarm')
    cbar = plt.colorbar(scat)
    cbar.set_label('Temperature, C')
    plt.xlabel('DO, mg/l')
    plt.ylabel('Conductivity, mS/cm')
    plt.show()
    plt.close()

    ####################################################
    ###### Make a PEST Version of Everything ###########
    ####################################################


    #read in Noa's PEST file
    pest_path = '/home/vpreston/Documents/field_work/pest/'
    pixhawk = base_path + 'pix_log17.gpx'
    suite = base_path + 'LOG99.txt'
    pest = pd.read_table(suite, delimiter=',', header=0, engine='c')
    pest.Year = pest.Year.values - 3
    pest.Month = pest.Month.values - 5
    pest.Day = pest.Day.values - 9
    pest.Hour = pest.Hour.values - 11
    pest = sensors.make_global_time(pest)
    pest.loc[:, 'Salinity'] = pest.apply(lambda x: calc_salinity(x['Temperature'], x['Conductivity']/1000), axis=1)
    print pest.Salinity.values
    sal = [np.round(x, 3) for x in pest.Salinity.values]
    np.savetxt('pest_salinity.txt', sal, fmt='%s')
    plt.scatter(pest.index, pest['Salinity'])
    plt.show()
    plt.close()

    pix = gpxpy.parse(open(pixhawk))
    track = pix.tracks[0]
    segment = track.segments[0]
    data = []
    for i, point in enumerate(segment.points):
    	data.append([point.longitude, point.latitude, point.elevation, point.time])

    pixdf = pd.DataFrame(data, columns=['Longitude', 'Latitude', 'Elevation', 'Time'])
    data = copy.copy(pixdf)
    data.loc[:, 'Year'] = data.apply(lambda x: x['Time'].year, axis=1)
    data.loc[:, 'Month'] = data.apply(lambda x: x['Time'].month, axis=1)
    data.loc[:, 'Day'] = data.apply(lambda x: x['Time'].day, axis=1)
    data.loc[:, 'Hour'] = data.apply(lambda x: x['Time'].hour, axis=1)
    data.loc[:, 'Minute'] = data.apply(lambda x: x['Time'].minute, axis=1)
    data.loc[:, 'Second'] = data.apply(lambda x: x['Time'].second, axis=1)
    data = sensors.make_global_time(data)
    pixdf = data

    dfs = [pest.drop_duplicates(subset='Julian_Date', keep='last').set_index('Julian_Date'),
           pixdf.drop_duplicates(subset='Julian_Date', keep='last').set_index('Julian_Date')]
    temp = pd.concat(dfs, axis=1, keys=['pest', 'pix'])
    inter_temp = temp.interpolate()
    df_index = dfs[1].index
    pest_df = inter_temp.loc[df_index]


    print pest_df.head(3)

    plt.scatter(pest_df['pix']['Longitude'], pest_df['pix']['Latitude'], c=pest_df['pest']['Salinity'], cmap='coolwarm')
    plt.show()
    plt.close()