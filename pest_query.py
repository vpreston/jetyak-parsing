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
    ###### Make a mission "analyzing" JetYak ###########
    ####################################################
    # Data to access
    # base_path = '/home/vpreston/Documents/IPP/jetyak_parsing/missions/falkor/'
    # miss = ['Falkor_0913.csv', 'Falkor_0914.csv', 'Falkor_0916.csv']
    # matplotlib.rcParams['figure.figsize'] = (15,15)
    # matplotlib.rcParams['font.size'] = 18
    # matplotlib.rcParams['figure.titlesize'] = 24
    # # matplotlib.rcParams['axes.grid'] = True
    # matplotlib.rcParams['axes.labelsize'] = 24
    # matplotlib.rcParams['legend.fontsize'] = 18
    # matplotlib.rcParams['grid.color'] = 'k'
    # matplotlib.rcParams['grid.linestyle'] = ':'
    # matplotlib.rcParams['grid.linewidth'] = 0.5

    # # Create mission operator
    # jy = jetyak.JetYak()
    # # jy.load_mission([base_path+m for m in miss], header=[0,1], meth_eff=0.1254)
    # # jy.save_mission(base_path, mission_name='trimmed')

    # # jy = jetyak.JetYak()
    # jy.load_mission([base_path+'trimmed_0.csv', base_path+'trimmed_2.csv'], header=0, simplify_mission=False)
