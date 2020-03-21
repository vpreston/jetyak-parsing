#!/usr/env/python

'''
Visualization library for JetYak mission dataframes.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import matplotlib.pyplot as plt
import numpy as np
import seawater.eos80 as gsw
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as mb
# from matplotlib.mlab import griddata
from sklearn.gaussian_process import GaussianProcess
from scipy.interpolate import griddata, Rbf
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import GPy
from shapely.geometry import LineString, Polygon, MultiLineString, Point
from shapely.ops import linemerge, unary_union, polygonize
from descartes import PolygonPatch
from scipy import stats

def compare_samples(jy, target='CH4', geo_epsilon=5.0, depth_epsilon=0.1, save_path=None):
    ''' Create plots that compare bottle samples with JetYak samples '''
    # header in form [('station', 'day', 'bottle_ch4_nM', 'bottle_co2_uatm', 'bottle_depth', 'lat', 'lon',
                    # 'jy_ch4_ppm', 'jy_ch4_uatm', 'jy_ch4_nm', jy_ch4_umolkg,
                    # 'jy_ch4_pstd', 'jy_ch4_ustd', 'jy_ch4_nstd', jy_ch4_umolstd,
                    # 'jy_co2_ppm', 'jy_co2_uatm', 'jy_co2_pstd', 'jy_co2_ustd',
                    # 'salinity', 'temperature', 'depth')]
    avg_samples, all_samples = jy.extract_bottle_locations(geo_epsilon=geo_epsilon, depth_epsilon=depth_epsilon, save_path=save_path)
    
    station_id = 1
    if target == 'CH4':
        labs = ['JetYak Raw Measurements, ppm', 'JetYak Measurements, uatm', 'JetYak Measurements, nM', 'JetYak Measurements, umolkg']
        bottle_id = 2
        jy_mid = [7, 8, 9, 10]
        jy_sid = [11, 12, 13, 14]
        xlabel = 'CH4 Bottle Measrements, nM'
    elif target == 'CO2':
        labs = ['JetYak Raw Measurements, ppm', 'JetYak Measurements, uatm']
        bottle_id = 3
        jy_mid = [15, 16]
        jy_sid = [17, 18]
        xlabel = 'pCO2 Bottle Measurements, uatm'

    title = 'Comparison with GeoEps '+str(geo_epsilon)+'m and DepthEps '+str(depth_epsilon)+'m'

    def make_plot(info, bottle_index, mean_index, std_index, xlabel, ylabel, title):
        ''' Method for making a plot from extracted bottle sample data '''
        plt.figure()
        labels = []
        dat_color = ['r', 'g', 'b', 'k', 'm', 'y']
        jydat = []
        sampdat = []
        for tup in info:
            if tup[3] == 0 or str(tup[3]) is 'nan':
                pass
            else:
                if str(tup[1]) not in labels:
                    plt.plot(float(tup[bottle_index]), tup[mean_index], c=dat_color[len(labels)], marker='o', label=str(tup[1]))
                    plt.errorbar(float(tup[bottle_index]), tup[mean_index], yerr=tup[std_index], c=dat_color[len(labels)])
                    labels.append(str(tup[1]))
                else:
                    plt.plot(float(tup[bottle_index]), tup[mean_index], c=dat_color[len(labels)-1], marker='o')
                    plt.errorbar(float(tup[bottle_index]), tup[mean_index], yerr=tup[std_index], c=dat_color[len(labels)-1])
                jydat.append(tup[mean_index])
                sampdat.append(tup[bottle_index])

        slope, intercept, r_value, p_value, std_err = stats.linregress(sampdat,jydat)
        line = [slope*x+intercept for x in sampdat]
        print(slope, intercept, r_value, p_value)
        plt.plot(sampdat, line)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
        plt.close()

    for i,j in zip(jy_mid, jy_sid): 
        make_plot(avg_samples, bottle_id, i, j, xlabel, labs[i-jy_mid[0]], title)

def st_plots(salt, temp, target, target_label, title, ax=None):
    ''' Casts plots in S-T space '''

    # Figure set-up
    # Copied from python-seawater ST plotting page
    # Figure bounds
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

    # Plot the data
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%1.0f') # Label every second level

    m = ax1.scatter(salt, temp, c=target, s=50, alpha=0.5, lw=0, cmap='viridis')
    ax1.set_xlabel('Salinity')
    ax1.set_ylabel('Temperature (C)')
    plt.title(title)
    cbar = plt.colorbar(m)
    cbar.set_label(target_label)

    return m

    # plt.show()
    # plt.close()

def val_depth_cascades(missions, depth_diff=0.5, limit=10.0, dates=('28 Jun', '29 Jun', '30 Jun', '01 Jul', '02 Jul', '04 Jul')):
    ''' aggregates data by depth and date '''
    layers = int((limit)/depth_diff)
    fig, ax = plt.subplots()

    for j,d in enumerate(missions):
        means_ch4 = []
        means_co2 = []
        depths = []
        for i in range(0, layers+1):
            lower = i*depth_diff
            upper = (i+1)*depth_diff
            temp_df = d[(d['Depth'] < upper) & (d['Depth'] >= lower)]
            means_ch4.append(np.mean(temp_df['CH4_ppm'].values))
            means_co2.append(np.mean(temp_df['CO2_uatm'].values))
            depths.append((lower+upper)/2)
        ax.plot(means_ch4, depths, label=dates[j], marker='o', linestyle='', alpha=1.0, markersize=10)

    ax.set_xlabel(r'Concentration of $CH_4$, $ppm$', fontsize=35)
    # ax.set_xlabel('Partial Pressure of $CO_2$, $\mu atm$', fontsize=35)
    ax.set_ylabel('Depth, $m$', fontsize=35)
    ax.invert_yaxis()
    ax.legend(fontsize=25)
    plt.show()
    plt.close()

def filled_contours(mission, region=None, buff=0.001, target='Depth', depth_lim=1.0, vmin=0.0, vmax=30.0):
    ''' Method for making a plot of target data (contours) onto a basemap object '''
    m = mission[(mission['Depth'] < depth_lim) & (mission['Depth'] > 0.1)]
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)

    #get bounding box
    if region is None:
        x_min = np.nanmin(m['Latitude']) - buff
        x_max = np.nanmax(m['Latitude']) + buff
        y_min = np.nanmin(m['Longitude']) - buff
        y_max = np.nanmax(m['Longitude']) + buff
    else:
        x_min = region[0] - buff
        x_max = region[1] + buff
        y_min = region[2] - buff
        y_max = region[3] + buff

    # make the map object
    base = Basemap(llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max,
                   resolution='l', projection='cyl')

    base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose=False)
    sf = shapefile.Reader('./cb.shp')
    for shape_rec in sf.shapeRecords():
        vertices = []
        codes = []
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)

    # make the grid object to project on
    proj_lon, proj_lat = base(*(m['Longitude'].values, m['Latitude'].values))
    num_cols, num_rows = 150, 150
    xi = np.linspace(y_min, y_max, num_cols)
    yi = np.linspace(x_min, x_max, num_rows)
    x1, y1 = np.meshgrid(xi, yi)

    #interpolate
    x, y, z = proj_lon, proj_lat, m[target].values
    z1 = griddata((x, y), z, (x1, y1), method='linear', rescale=True)

    # rbf = Rbf(x[0::100], y[0::100], z[0::100], method='linear')
    # z1 = rbf(x1, y1) + np.mean(m[target].values)

    # gp = GaussianProcess()#theta0=0.1, thetaL=.001, thetaU=1., nugget=0.01)
    # gp.fit(X=np.column_stack([x[0::100],y[0::100]]), y=z[0::100])
    # z1 = gp.predict(np.column_stack([x1.flatten(), y1.flatten()])).reshape(x1.shape)

    # weighted, _, _ = np.histogram2d(x, y, weights=z, normed=False, bins=1000)
    # count, xedges, yedges = np.histogram2d(x,y,bins=1000)
    # z1 = weighted/count
    
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # x1, y1 = np.meshgrid(xcenters, ycenters)

    # plt.pcolormesh(x, y, z, cmap='RdBu', vmin=vmin, vmax=vmax)

    # con = base.contourf(x1, y1, z1.T, cmap=plt.cm.bwr)

    # xvals = np.array([[q, r] for q, r in zip(x, y)])
    # zvals = np.reshape(np.array(z), (len(z), 1))

    # # kern = GPy.kern.sde_Matern52(input_dim=2, lengthscale=1.0, variance=np.var(z))
    # # kern = GPy.kern.sde_Exponential(input_dim=2, lengthscale=10.0, variance=np.var(z))
    # kern = GPy.kern.sde_RBF(input_dim=2, lengthscale=1.0, variance=np.var(z))
    # # kern = GPy.kern.sde_RatQuad(input_dim=2, lengthscale=0.001, variance=np.var(z))

    # mod = GPy.models.GPRegression(xvals[0::5], zvals[0::5], kern)
    # print 'initializing'
    # mod.initialize_parameter()
    # mod.optimize_restarts(num_restarts=2, messages=True, robust=True)
    # print kern

    # data = np.vstack([x1.ravel(), y1.ravel()]).T
    # observations, var = mod.predict(data, full_cov=False, include_likelihood=True)
    # observations = observations.reshape(x1.shape) + np.mean(m[target].values)
    # obs = observations
    con = base.contourf(x1, y1, z1, zorder=10, alpha=1.0, cmap='viridis', levels=np.linspace(vmin, vmax, 15))
    # # con = base.contourf(x1, y1, obs, cmap = 'viridis', levels=np.linspace(np.nanmin(obs), np.nanmax(obs), 15))
    for contour in con.collections:
        contour.set_clip_path(clip)

    # base.scatter(x, y, c=z, cmap='viridis', vmin=np.nanmin(z1), vmax=np.nanmax(z1), edgecolors='face', linewidths=0)

    cbar = plt.colorbar(con)
    cbar.set_label(target, fontsize=30)
    cbar.ax.tick_params(labelsize=20)
    # base.scatter(x,y,zorder=5,s=0.1)
    plt.show()
    plt.close()

def partial_contours(mission, region, target='Depth', depth_lim=1.0, vmin=0.0, vmax=30.0):
    ''' Method for making a plot of target data (contours) onto a basemap object '''

    # make the map object
    #throw away junk and values out of range
    m = mission[(mission['Depth'] < depth_lim) & (mission['Depth'] > 0.25)]

    x_min = np.nanmin(m['Latitude']) - 0.001
    x_max = np.nanmax(m['Latitude']) + 0.001
    y_min = np.nanmin(m['Longitude']) - 0.001
    y_max = np.nanmax(m['Longitude']) + 0.001

    base = Basemap(llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max,
                   resolution='l', projection='cyl')

    allx, ally = base(*(m['Longitude'].values, m['Latitude'].values))
    m.loc[:,'proj_lon'] = allx
    m.loc[:,'proj_lat'] = ally

    # make the grid object to project on for each region
    buff = 0.0003
    for r in region:
        tmp = m[(m['proj_lat'] >= r[0]-buff) & (m['proj_lat'] <= r[1]+buff) & (m['proj_lon'] >= r[2]-buff) & (m['proj_lon'] <= r[3]+buff)]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        num_cols, num_rows = 500, 500
        xi = np.linspace(r[2]-buff, r[3]+buff, num_cols)
        yi = np.linspace(r[0]-buff, r[1]+buff, num_rows)
        x1, y1 = np.meshgrid(xi, yi)

        #interpolate
        x, y, z = tmp['proj_lon'].values, tmp['proj_lat'].values, tmp[target].values
        z1 = griddata((x, y), z, (x1, y1), method='linear', rescale=True) 

        base.contourf(x1, y1, z1, zorder=4, alpha=1.0, cmap='viridis', levels=np.linspace(vmin, vmax, 15))
        cbar = plt.colorbar()
        cbar.set_label(target)
        cbar.ax.tick_params(labelsize=40) 
        base.scatter(tmp['proj_lon'].values, tmp['proj_lat'].values, zorder=5,s=0.1)
        ax.set_xlim([r[2]-buff, r[3]+buff])
        ax.set_ylim([r[0]-buff, r[1]+buff])
        plt.show()
        plt.close()

def colored_scatter(mission, target='Depth', depth_lim=1.0, vmin=0.0, vmax=30.0):
    ''' Method for making a plot of target data (contours) onto a basemap object '''
    m = mission[(mission['Depth'] < depth_lim) & (mission['Depth'] > 0.25) & (mission['Salinity'] > 25.)]
    thin_line = mission[(mission['Depth'] >= depth_lim) & (mission['Salinity'] > 25.)]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #get bounding box
    x_min = np.nanmin(m['Latitude']) - 0.001
    x_max = np.nanmax(m['Latitude']) + 0.001
    y_min = np.nanmin(m['Longitude']) - 0.001
    y_max = np.nanmax(m['Longitude']) + 0.001

    print(x_min, x_max, y_min, y_max)
    # make the map object
    base = Basemap(llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max,
                   resolution='l', projection='cyl')

    # base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
    proj_lon, proj_lat = base(*(m['Longitude'].values, m['Latitude'].values))
    path_lon, path_lat = base(*(thin_line['Longitude'].values, thin_line['Latitude'].values))

    # np.savetxt('0913.txt', np.array([proj_lon, proj_lat]).T)
    plume_lon, plume_lat = np.loadtxt('missions/falkor/updated_plumes.txt', delimiter=',', unpack=True)

    scat = base.scatter(proj_lon, proj_lat, zorder=5, s=30.0, alpha=0.5, c=m[target].values, cmap='coolwarm', lw=0, vmin=vmin, vmax=vmax)
    base.scatter(path_lon, path_lat, zorder=5, s=0.3, alpha=0.3, c="0.5")
    base.scatter(-1*plume_lon, plume_lat, s=100, c='r', lw=0, zorder=6)
    #9/11
    # casts = [(44.2029833, -124.8509833),
    #          (44.3626167, -124.1628333),
    #          (44.3702167, -124.1846000)]

    #9/16
    # casts = [(44.4563167, -124.2660000)]

    cbar = plt.colorbar(scat)
    # base.scatter([x[1] for x in casts], [x[0] for x in casts], c='r', s=100., lw=0)
    plt.show()
    plt.close()

def regional_comparison(missions, regions, depth_diff=0.5, limit=10.0):
    ''' Method to create val-depth cascades for different regions in the river '''
    divided_missions = []
    poly = None
    edges = []
    sf = shapefile.Reader('./cb.shp')
    for shape_rec in sf.shapeRecords():
        pts = shape_rec.shape.points
        poly = Polygon(pts)

    for s in regions.shapeRecords():
        pts = s.shape.points
        edges.append(LineString(pts))

    merged = linemerge([poly.boundary, edges[0], edges[1], edges[2]])
    borders = unary_union(merged)
    polygons = polygonize(borders)


    # for p in polygons:

    # # cut a polygon into pieces with these lines

    #     for m in missions:
    #         temp = add_poly(m, p)
    #         temp = temp[(temp['in_poly'] == True)]
    #         divided_missions.append(temp)
    #     val_depth_cascades(divided_missions, depth_diff=depth_diff, limit=limit)
    #     divided_missions = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for p in polygons:
        ring_patch = PolygonPatch(p)
        ax.add_patch(ring_patch)
        ax.scatter(missions[0]['Longitude'], missions[0]['Latitude'],alpha=0)
    plt.show()
    plt.close()


def add_poly(df, polygon):
    df.loc[:, 'in_poly'] = df.apply(lambda x: polygon.contains(Point(x['Longitude'], x['Latitude'])),axis=1)
    return df

