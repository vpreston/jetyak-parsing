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
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import shapefile
import GPy
from shapely.geometry import LineString, Polygon, MultiLineString, Point
from shapely.ops import linemerge, unary_union, polygonize
from descartes import PolygonPatch

def compare_samples(jy, geo_epsilon=5.0, depth_epsilon=0.1, save_path=None):
    ''' Create plots that compare bottle samples with JetYak samples '''
    # header in form [('station', 'day', 'bottle_ch4_nM', 'bottle_co2_uatm', 'bottle_depth', 'lat', 'lon',
                    # 'jy_ch4_ppm', 'jy_ch4_uatm', 'jy_ch4_nm', 'jy_ch4_pstd', 'jy_ch4_ustd', 'jy_ch4_nstd',
                    # 'jy_co2_ppm', 'jy_co2_uatm', 'jy_co2_pstd', 'jy_co2_ustd',
                    # 'salinity', 'temperature', 'depth')]
    avg_samples, all_samples = jy.extract_bottle_locations(geo_epsilon=geo_epsilon, depth_epsilon=depth_epsilon, save_path=save_path)
    labs = ['JetYak Raw Measurements, ppm', 'JetYak Measurements, uatm', 'JetYak Measurements, nM']
    labs = ['JetYak Raw Measurements, ppm', 'JetYak Measurements, uatm']

    title = 'Comparison with GeoEps '+str(geo_epsilon)+'m and DepthEps '+str(depth_epsilon)+'m'

    def make_plot(info, index, ylabel, title):
        ''' Method for making a plot from extracted bottle sample data '''
        plt.figure()
        labels = []
        dat_color = ['r', 'g', 'b', 'k', 'm', 'y']
        for tup in info:
            if tup[3] == 0 or str(tup[3]) is 'nan':
                pass
            else:
                if str(tup[1]) not in labels:
                    plt.plot(float(tup[3]), tup[index], c=dat_color[len(labels)], marker='o', label=str(tup[1]))
                    plt.errorbar(float(tup[3]), tup[index], yerr=tup[index+2], c=dat_color[len(labels)])
                    labels.append(str(tup[1]))
                else:
                    plt.plot(float(tup[3]), tup[index], c=dat_color[len(labels)-1], marker='o')
                    plt.errorbar(float(tup[3]), tup[index], yerr=tup[index+2], c=dat_color[len(labels)-1])
        plt.xlabel('pCO2 Bottle Measurements, uatm')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
        plt.close()

    for i in [13, 14]:
        make_plot(avg_samples, i, labs[i-13], title)

def st_plots(salt, temp, target, target_label, title):
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

    plt.show()
    plt.close()

def val_depth_cascades(missions, depth_diff=0.5, limit=10.0):
    ''' aggregates data by depth and date '''
    dates = ['28 Jun', '29 Jun', '30 Jun', '01 Jul', '02 Jul', '04 Jul']

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

def filled_contours(mission):
    ''' Method for making a plot of target data (contours) onto a basemap object '''
    m = mission[1][(mission[1]['Depth'] < 1.0)]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #get bounding box
    x_min = np.nanmin(m['Latitude']) - 0.01
    x_max = np.nanmax(m['Latitude']) + 0.01
    y_min = np.nanmin(m['Longitude']) - 0.01
    y_max = np.nanmax(m['Longitude']) + 0.01

    # make the map object
    base = Basemap(llcrnrlon=y_min, llcrnrlat=x_min, urcrnrlon=y_max, urcrnrlat=x_max,
                   resolution='l', projection='cyl')

    base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose= True)
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
    num_cols, num_rows = 100, 100
    xi = np.linspace(np.nanmin(proj_lon), np.nanmax(proj_lon), num_cols)
    yi = np.linspace(np.nanmin(proj_lat), np.nanmax(proj_lat), num_rows)
    x1, y1 = np.meshgrid(xi, yi)

    #interpolate
    x, y, z = proj_lon, proj_lat, m['CH4_ppm'].values
    # z1 = griddata((x, y), z, (x1, y1), method='linear', rescale=True)

    # weighted, _, _ = np.histogram2d(x, y, weights=z, normed=False, bins=100)
    # count, xedges, yedges = np.histogram2d(x,y,bins=100)
    # z1 = weighted/count
    
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # x1, y1 = np.meshgrid(xcenters, ycenters)

    # con = base.contourf(x1, y1, z1.T, cmap=plt.cm.bwr)

    xvals = np.array([[m, n] for m, n in zip(x, y)])
    zvals = np.reshape(np.array(z), (len(z), 1))

    # kern = GPy.kern.sde_Matern52(input_dim=2, lengthscale=0.001, variance=np.var(z))
    kern = GPy.kern.sde_Exponential(input_dim=2, lengthscale=0.001, variance=np.var(z))
    # kern = GPy.kern.sde_RatQuad(input_dim=2, lengthscale=0.001, variance=np.var(z))

    mod = GPy.models.GPRegression(xvals[0::5], zvals[0::5], kern)
    print 'initializing'
    mod.initialize_parameter()
    mod.optimize_restarts(num_restarts=2, messages=True, robust=True)
    print kern

    data = np.vstack([x1.ravel(), y1.ravel()]).T
    obs, var = mod.predict(data, full_cov=False, include_likelihood=True)

    # con = base.contourf(x1, y1, z1, zorder=4, alpha=0.6, cmap='RdPu')
    con = base.contourf(x1, y1, obs.reshape(x1.shape), cmap = 'viridis', levels=np.linspace(np.nanmin(obs), np.nanmax(obs), 15))
    for contour in con.collections:
        contour.set_clip_path(clip)

    # base.scatter(x, y, c=z, cmap='viridis', vmin=np.nanmin(obs), vmax=np.nanmax(obs), edgecolors='face', linewidths=0)

    cbar = plt.colorbar(con)
    cbar.set_label('CH4 (ppm)')
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
