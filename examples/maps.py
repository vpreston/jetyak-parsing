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
    #69.121595, -105.019215
    base = Basemap(llcrnrlon=-170, llcrnrlat=0, urcrnrlon=-30, urcrnrlat=80,
                   resolution='l', projection='merc', suppress_ticks=True)

    # base = Basemap(llcrnrlon=-120, llcrnrlat=68, urcrnrlon=-100, urcrnrlat=74,
    #                resolution='h', projection='merc', suppress_ticks=True)


    # base.arcgisimage(service='World_Topo_Map', xpixels=1500, verbose=True)
    base.drawcoastlines()
    base.drawcountries()
    # base.drawlakes()
    # base.fillcontinents(color='coral',lake_color='aqua')
    # base.drawlsmask(land_color='coral', ocean_color='aqua', lakes=True)
    # base.drawparallels(np.arange(-80.,81.,2.),labels=[True,True,False,False],dashes=[2,2],color='white')
    # base.drawmeridians(np.arange(-180.,181.,10.),labels=[True,True,True,False],dashes=[2,2],color='white')
    # base.drawmapboundary(fill_color='aqua')
    # base.drawrivers(linewidth=0.5, linestyle='solid', color='blue')
    base.drawparallels(np.arange(-90.,91.,10.),labels=[True,True,False,False],dashes=[2,2],color='white')
    base.drawmeridians(np.arange(-180.,181.,30.),labels=[False,False,False,True],dashes=[2,2],color='white')
    base.drawparallels(np.arange(66.,67., 100.),labels=[False,False,False,True],dashes=[2,2],color='red')
    base.drawstates(linewidth=2., color='grey')
    base.bluemarble()
    plt.show()

    # base.scatter(dock_reference[1], dock_reference[0], s=500, marker='*', label='Freshwater Creek Mouth', zorder=10, edgecolor='k', facecolor='r')
    # for radius in [500*i for i in range(10)]:
    #     lats, lons = getCircle(dock_reference[0], dock_reference[1], radius)
    #     base.plot(lons, lats, c='grey')
    #     if radius == 0:
    #         pass
    #         # plt.gca().annotate('Embayment', xy=(lons[270], lats[270]+0.001), xytext=(lons[270]+0.0005, lats[270]+0.002), fontsize=22, ha='center')
    #         # plt.gca().annotate('Freshwater Creek Mouth', xy=(lons[270], lats[270]+0.0005), fontsize=10, ha='right')
    #     else:
    #         plt.gca().annotate(str(radius)+'m', xy=(lons[270], lats[270]+0.0003), fontsize=22, ha='center')

    # colors = np.flip(plt.cm.viridis(np.linspace(0,1,5)), axis=0)
    # for i, m in enumerate(jy.mission[0:5]):
    #     base.scatter(m['Longitude'], m['Latitude'], label=date_labels[i], s=1, c=colors[i], zorder=10-i, lw=0)

    # lgnd = plt.legend(loc='upper left')
    # for handle in lgnd.legendHandles[1:]:
    #     handle.set_sizes([200])

    # ax = plt.gca()
    # def xformat(x, pos=None): return lon2str(x)
    # def yformat(x, pos=None): return lat2str(x)
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xformat))
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yformat))

    # plt.show()
    # plt.close()