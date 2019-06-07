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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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


    # # Can now perform work with the sensors
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
    matplotlib.rcParams['font.size'] = 18
    matplotlib.rcParams['figure.titlesize'] = 24
    # matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.labelsize'] = 24
    matplotlib.rcParams['legend.fontsize'] = 18
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5

    # Create mission operator
    jy = jetyak.JetYak()
    # jy.load_mission([base_path+m for m in miss], header=[0,1], meth_eff=0.15)
    # jy.save_mission(base_path, mission_name='trimmed')

    # jy = jetyak.JetYak()
    jy.load_mission([base_path+'trimmed_0.csv', base_path+'trimmed_2.csv'], header=0, simplify_mission=False)


    target='CH4_nM'
    target_constant = 1

    for m in jy.mission[1]: #Stonewall Bank
        # select which part of the watercolumn to analyze
        tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)] #surface
        ctmp = m[m['Depth'] > 0.5] #water column

        # plotting helpers
        vmin = np.nanmin(tmp[target].values)
        vmax = np.nanmax(tmp[target].values)
        x_min = np.nanmin(tmp['Latitude']) - 0.001
        x_max = np.nanmax(tmp['Latitude']) + 0.001
        y_min = np.nanmin(tmp['Longitude']) - 0.001
        y_max = np.nanmax(tmp['Longitude']) + 0.001

        #generate histograms
        # plt.hist(tmp[target].values, 50, range=(vmin, vmax), density=False, log=True)
        # plt.xlabel('Methane, nM')
        # plt.ylabel('Log Sample Count')
        # plt.axvspan(2.7, vmax, alpha=0.1, color='red')
        # plt.xlim(vmin, vmax)
        # plt.show()
        # plt.close()

        #generate joint plots
        j = sns.jointplot(ctmp['CH4_nM'], ctmp['Depth'], height=15, ratio=3, s=10, marginal_kws=dict(bins=25, kde=True))
        j.set_axis_labels('Methane, nM', 'Depth, m')
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()

        #generate kde plots
        j = sns.distplot(tmp['CH4_nM'], label='Surface Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(tmp['CH4_nM']), np.nanmax(tmp['CH4_nM']))))
        sns.distplot(ctmp['CH4_nM'], label='All Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(ctmp['CH4_nM']), np.nanmax(ctmp['CH4_nM']))))
        plt.legend()
        plt.show()
        plt.close()

        #generate 3D plots
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, projection='3d')
        # p = np.ones(len(tmp['Longitude']))*np.nanmin(tmp['CH4_nM'])
        # # ax1.bar3d(tmp['Longitude'], tmp['Latitude'], np.zeros(len(tmp['Longitude'])), 0.0001, 0.0001, tmp['CH4_nM'])
        # ax1.scatter3D(ctmp['Longitude'], ctmp['Latitude'], ctmp['CH4_nM'], c=ctmp['CH4_nM'], cmap='coolwarm')
        # # ax1.set_zlim(np.nanmax(ctmp['Depth']), np.nanmin(ctmp['Depth']))
        # ax1.set_zlim(np.nanmin(ctmp['CH4_nM']), np.nanmax(ctmp['CH4_nM']))
        # plt.show()
        # plt.close()

        #generate slices
        scat = plt.scatter(ctmp.index-ctmp.index[0], ctmp['Depth'], c=ctmp['CH4_nM'], cmap='coolwarm', s=0.5, norm=matplotlib.colors.LogNorm())
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(scat)
        cbar.set_label('Methane, nM')
        plt.xlabel('Sample Number')
        plt.ylabel('Depth, m')
        plt.show()
        plt.close()

        #generate thresholded surface plots
        m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        base = Basemap(llcrnrlon=y_min,
                       llcrnrlat=x_min,
                       urcrnrlon=y_max,
                       urcrnrlat=x_max,
                       resolution='l',
                       projection='cyl',
                       suppress_ticks=False)
        proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        scat = base.scatter(path_lon,
                            path_lat,
                            zorder=5,
                            s=30.0,
                            alpha=1.0,
                            c=m_tmp[target].values,
                            cmap='coolwarm',
                            lw=0)
        base.scatter(proj_lon,
                     proj_lat,
                     zorder=3,
                     s=0.3,
                     alpha=0.1,
                     c='k')
        base.scatter(-1*plume_lon, plume_lat, s=200, c='m', lw=0, zorder=6, label='Bubble plumes')
        base.scatter([x[1] for x in casts], [x[0] for x in casts], s=300, c='k', marker='*', zorder=6, label='CTD025')
        cbar = plt.colorbar(scat)
        cbar.set_label('Methane, nM')
        ax = plt.gca()
        def xformat(x, pos=None): return lon2str(x)
        def yformat(x, pos=None): return lat2str(x)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        print 'length ', length

        base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        ax.set_xlabel('')

        # plt.xticks(rotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.legend()
        # plt.gcf().subplots_adjust(bottom=0.5)
        plt.gcf().tight_layout(rect=(0,0.11,1,1))
        plt.show()
        plt.close()

        #Create cascades
        target = 'CH4_nM' #'Temperature' #'Salinity' #'CO2_uatm' #'CH4_umolkg'
        avgs = []
        ds = []
        depths = [0.5+i*0.10 for i in range(0,100)]
        last_depth = depths[0]


        for d in depths:
            tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
            avgs.append(np.mean(tmp[target]))
            ds.append(np.mean(tmp['Depth']))
            last_depth = d
        plt.scatter(avgs, ds)
        avgs = []
        ds = []

        plt.ylabel('Depth, m')
        plt.gca().invert_yaxis()
        plt.xlabel('Methane, nM')
        # plt.xlabel('pCO2, $\mu$atm')
        # plt.xlabel('Salinity, PSS')
        # plt.xlabel('Temperature, C')
        plt.show()
        plt.close()

for m in jy.mission[0]:
        # select which part of the watercolumn to analyze
        tmp = m[(m['Depth'] < 1.5) & (m['Depth'] > 0.50)] #surface
        ctmp = m[m['Depth'] > 0.5] #water column

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
        j = sns.jointplot(ctmp['CH4_nM'], ctmp['Depth'], height=15, ratio=3, s=10, marginal_kws=dict(bins=25, kde=True))
        j.set_axis_labels('Methane, nM', 'Depth, m')
        plt.gca().invert_yaxis()
        plt.show()
        plt.close()

        #generate kde plots
        j = sns.distplot(tmp['CH4_nM'], label='Surface Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(tmp['CH4_nM']), np.nanmax(tmp['CH4_nM']))))
        sns.distplot(ctmp['CH4_nM'], label='All Observations', kde=False, bins=50, hist_kws=dict(log=True, range=(np.nanmin(ctmp['CH4_nM']), np.nanmax(ctmp['CH4_nM']))))
        plt.legend()
        plt.show()
        plt.close()

        #generate slices
        scat = plt.scatter(ctmp.index-ctmp.index[0], ctmp['Depth'], c=ctmp['CH4_nM'], cmap='coolwarm', s=0.5, norm=matplotlib.colors.LogNorm())
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(scat)
        cbar.set_label('Methane, nM')
        plt.xlabel('Sample Number')
        plt.ylabel('Depth, m')
        plt.show()
        plt.close()

        #generate thresholded surface plots
        m_tmp = ctmp[ctmp['CH4_nM'] >= 2.7]
        tmp = ctmp[ctmp['CH4_nM'] < 2.7]
        plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)
        casts = [(44.2029833, -124.8509833),(44.3626167, -124.1628333),(44.3702167, -124.1846000), (44.4563167, -124.2660000)]

        base = Basemap(llcrnrlon=y_min,
                       llcrnrlat=x_min,
                       urcrnrlon=y_max,
                       urcrnrlat=x_max,
                       resolution='l',
                       projection='cyl',
                       suppress_ticks=False)
        proj_lon, proj_lat = base(*(tmp['Longitude'].values, tmp['Latitude'].values))
        path_lon, path_lat = base(*(m_tmp['Longitude'].values, m_tmp['Latitude'].values))

        scat = base.scatter(path_lon,
                            path_lat,
                            zorder=5,
                            s=30.0,
                            alpha=1.0,
                            c=m_tmp[target].values,
                            cmap='coolwarm',
                            lw=0)
        base.scatter(proj_lon,
                     proj_lat,
                     zorder=3,
                     s=0.3,
                     alpha=0.1,
                     c='k')
        base.scatter(-1*plume_lon, plume_lat, s=200, c='m', lw=0, zorder=6, label='Bubble plumes')
        base.scatter([x[1] for x in casts], [x[0] for x in casts], s=300, c='k', marker='*', zorder=6, label='CTD025')
        cbar = plt.colorbar(scat)
        cbar.set_label('Methane, nM')
        ax = plt.gca()
        def xformat(x, pos=None): return lon2str(x)
        def yformat(x, pos=None): return lat2str(x)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

        length = get_distance((x_min+0.001, y_min+0.001), (x_min+0.001, y_min+0.00226))
        print 'length ', length

        base.plot([y_min+0.001, y_min+0.00226], [x_max-0.001, x_max-0.001], marker='|', c='k')
        ax.annotate(str(int(length))+'m', (np.mean([y_min+0.001, y_min+0.00226])-0.0005, x_max-0.0014), fontsize=14)
        ax.set_xlabel('')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.legend()
        plt.gcf().tight_layout(rect=(0,0.11,1,1))
        plt.show()
        plt.close()

        #Create cascades
        target = 'CH4_nM'
        avgs = []
        ds = []
        depths = [0.5+i*0.10 for i in range(0,100)]
        last_depth = depths[0]

        for d in depths:
            tmp = m[(m['Depth'] <= d) & (m['Depth'] > last_depth)]
            avgs.append(np.mean(tmp[target]))
            ds.append(np.mean(tmp['Depth']))
            last_depth = d
        plt.scatter(avgs, ds)
        plt.ylabel('Depth, m')
        plt.gca().invert_yaxis()
        plt.xlabel('Methane, nM')
        plt.show()
        plt.close()
