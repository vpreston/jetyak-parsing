#!/usr/bin/env python

'''
Creates JetYak Class which contains various sensors, and allows for operations between sensors.

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import sensors
import pandas as pd
import numpy as np
import copy
import utm
from gasex import sol

class JetYak(object):
    ''' Class which allows for sensor interfaces and basic queries to be made about a jetyak mission '''
    def __init__(self, bounds=None, trim_vals=None, args=None):
        ''' A list of filepaths for the sensors deployed on the mission provided '''

        self.ctd = None
        self.suna = None
        self.optode = None
        self.gga = None
        self.airmar = None
        self.bottle_samples = None
        self.mission = []
        self.bounds = bounds
        self.trim_vals = trim_vals
        self.sensors = []
        self.sensor_names = []
        if args is not None:
            self.offset = args[0]

    def attach_sensor(self, sensor, dirs):
        ''' Method to add sensors for parsing and cleaning on the Jetyak '''
        if 'ctd' in sensor:
            print 'Attaching CTD'
            self.ctd = sensors.CTD(dirs, self.bounds, self.trim_vals)
            self.ctd.clean_ctd()
            self.sensors.append(self.ctd)
            self.sensor_names.append('ctd')
        elif 'gga' in sensor:
            print 'Attaching GGA'
            self.gga = sensors.GGA(dirs, self.bounds, self.trim_vals)
            self.gga.clean_gga()
            self.sensors.append(self.gga)
            self.sensor_names.append('gga')
        elif 'airmar' in sensor:
            print 'Attaching AirMar'
            self.airmar = sensors.AirMar(dirs, self.bounds, self.trim_vals)
            self.airmar.clean_airmar()
            self.sensors.append(self.airmar)
            self.sensor_names.append('airmar')
        elif 'optode' in sensor:
            print 'Attaching Optode'
            self.optode = sensors.Optode(dirs, self.bounds, self.trim_vals)
            self.optode.set_characteristics(offset=self.offset)
            self.optode.clean_optode()
            self.sensors.append(self.optode)
            self.sensor_names.append('optode')
        else:
            print 'Only supporting CTD, GGA, Optode, and Airmar inputs \
                       at this time.'

    def create_mission(self, args):
        '''Method to combine (geo associate and time associate) all valid sensor signals.
        Args is a dictionary of values which may be useful in processing some mission data
        geoframe (string): which sensor to use for lat, lon
        geolabels (tuple strings): labels to use for the chosen geoframe'''

        # get the index of the geoframe for interpolating
        ind = self.sensor_names.index(args['geoframe'])

        # kill duplicate timestamps if they exist
        df = []
        for s in self.sensors:
            temp = s.get_df().drop_duplicates(subset='Julian_Date', keep='last').set_index('Julian_Date')
            df.append(temp)

        # create meta dataframe and perform interpolation on geoframe
        all_temp = pd.concat(df, axis=1, keys=self.sensor_names)
        inter_temp = all_temp.interpolate()
        df_index = df[ind].index
        self.mission.append(inter_temp.loc[df_index])

    def save_mission(self, save_path, mission_name):
        '''Method to save sensors and mission files'''
        # save sensors first
        for n, s in zip(self.sensor_names, self.sensors):
            s.get_df().to_csv(save_path+n+'.csv')

        # save the mission
        if len(self.mission) > 0:
            for i, m in enumerate(self.mission):
                # m = m.dropna(axis=1)
                if 'trimmed' in mission_name:
                    m.to_csv(save_path+mission_name+'_'+str(i)+'.csv')
                else:
                    m.to_csv(save_path+mission_name)

    def load_mission(self, mission_path, header=0, simplify_mission=True, meth_eff=0.03, carb_eff=0.70):
        '''Method to load previously cleaned and collated mission into working memory
        simplify_mission is a boolean flag for whether to store the loaded mission in a
        smaller working memory format'''
        # mission is a list of dataframes
        self.mission = []
        for path in mission_path:
            temp = pd.read_table(path, delimiter=',', header=header)
            if not simplify_mission:
                self.mission.append(temp)
            else:
                self.mission.append(strip_mission(temp, meth_eff=meth_eff, carb_eff=carb_eff))

    def add_bottle_samples(self, file_path):
        ''' Method to add bottle samples taken in parallel to the jetyak mission '''
        self.bottle_samples = clean_samples(file_path)

    def collapse_bottle_samples(self):
        '''Method to collapse the bottle sample frame to kill triplicates'''
        df = self.bottle_samples

        cstats = []
        cdepth = []
        clat = []
        clon = []
        cch4_mean = []
        cch4_std = []
        cco2_mean = []
        cco2_std = []
        cmonth = []
        cday = []

        unique_stations = np.unique(df['station'].values)
        for st_id in unique_stations:
            #get the station
            temp = df[df['station'] == st_id]
            depth_id = np.unique(temp['depth'].values)
            for d in depth_id:
                extracted = temp[temp['depth'] == d]
                cstats.append(st_id)
                cdepth.append(d)
                clat.append(extracted['lat'].values[0])
                clon.append(extracted['lon'].values[0])
                cch4_mean.append(np.mean(extracted['[CH4] nM'].values))
                cch4_std.append(np.std(extracted['[CH4] nM'].values))
                cco2_mean.append(np.mean(extracted['pCO2'].values))
                cco2_std.append(np.std(extracted['pCO2'].values))
                cmonth.append(extracted['month'].values[0])
                cday.append(extracted['day'].values[0])

        collapsed_df = pd.DataFrame()
        collapsed_df.loc[:, 'station'] = cstats
        collapsed_df.loc[:, 'depth'] = cdepth
        collapsed_df.loc[:, 'lat'] = clat
        collapsed_df.loc[:, 'lon'] = clon
        collapsed_df.loc[:, 'ch4_mean'] = cch4_mean
        collapsed_df.loc[:, 'ch4_std'] = cch4_std
        collapsed_df.loc[:, 'co2_mean'] = cco2_mean
        collapsed_df.loc[:, 'co2_std'] = cco2_std
        collapsed_df.loc[:, 'month'] = cmonth
        collapsed_df.loc[:, 'day'] = cday

        return collapsed_df


    def match_bottles(self, cdf, geo_epsilon=10.0, depth_epsilon=0.1):
        '''Method to match the collapsed bottle samples to JetYak obseervations'''
        match_df = copy.copy(cdf)

        #there should be no duplicates, so let's just run through the dataframe
        jch4_mean = []
        jch4_std = []
        jco2_mean = []
        jco2_std = []
        jsal_mean = []
        jsal_std = []
        jtemp_mean = []
        jtemp_std = []

        missions = []

        for m in self.mission:
            m.loc[:, 'utmlat'] = m.apply(lambda x: convert_to_utm((x['Latitude'], x['Longitude']))[0], axis=1)
            m.loc[:, 'utmlon'] = m.apply(lambda x: convert_to_utm((x['Latitude'], x['Longitude']))[1], axis=1)
            missions.append(m)

        for i in range(len(cdf.index)):
            entry = cdf.iloc[i]
            day = entry['day']
            if day == 28:
                m = missions[0]
            elif day == 29:
                m = missions[1]
            elif day == 30:
                m = missions[2]
            elif day == 1:
                m = missions[3]
            elif day == 2:
                m = missions[4]
            else:
                m = None

            if entry['depth'] == 0.75:
                d = entry['depth']-0.15
                de = 0.36
            elif entry['depth'] > 0.75:
                d = entry['depth']
                de = 0.36
            else:
                d = entry['depth']
                de = depth_epsilon

            if m is None:
                jch4_mean.append(None)
                jch4_std.append(None)
                jco2_mean.append(None)
                jco2_std.append(None)
                jsal_mean.append(None)
                jsal_std.append(None)
                jtemp_mean.append(None)
                jtemp_std.append(None)
            else:
                entry_lat, entry_lon = convert_to_utm((entry['lat'], entry['lon']))
                temp = m[(m['Depth'] <= d + de) & (m['Depth'] >= d-de)]
                temp = temp[(((temp['utmlat']-entry_lat)**2 + (temp['utmlon']-entry_lon)**2) <= geo_epsilon**2)]
                # temp = temp[(temp['Latitude']-entry['lat'])**2 + (temp['Longitude']-entry['lon'])**2 <= geo_epsilon**2]
                jch4_mean.append(np.mean(temp['CH4_nM'].values))
                jch4_std.append(np.std(temp['CH4_nM'].values))
                jco2_mean.append(np.mean(temp['CO2_uatm'].values))
                jco2_std.append(np.std(temp['CO2_uatm'].values))
                jsal_mean.append(np.mean(temp['Salinity'].values))
                jsal_std.append(np.std(temp['Salinity'].values))
                jtemp_mean.append(np.mean(temp['Temperature'].values))
                jtemp_std.append(np.std(temp['Temperature'].values))

        match_df.loc[:, 'jch4_mean'] = jch4_mean
        match_df.loc[:, 'jch4_std'] = jch4_std
        match_df.loc[:, 'jco2_mean'] = jco2_mean
        match_df.loc[:, 'jco2_std'] = jco2_std
        match_df.loc[:, 'jsal_mean'] = jsal_mean
        match_df.loc[:, 'jsal_std'] = jsal_std
        match_df.loc[:, 'jtemp_mean'] = jtemp_mean
        match_df.loc[:, 'jtemp_std'] = jtemp_std

        return match_df

    def extract_bottle_locations(self, geo_epsilon=10.0, depth_epsilon=0.1, save_path=None):
        '''Method to create dataset which matches jetyak and bottle sample information.
        Saves to file is filepath is provided'''
        compare = [('station', 'day', 'bottle_ch4_nM', 'bottle_co2_uatm', 'bottle_depth', 'lat', 'lon',
                    'jy_ch4_ppm', 'jy_ch4_uatm', 'jy_ch4_nm', 'jy_ch4_umolkg',
                    'jy_ch4_pstd', 'jy_ch4_ustd', 'jy_ch4_nstd', 'jy_ch4_umolstd',
                    'jy_co2_ppm', 'jy_co2_uatm', 'jy_co2_pstd', 'jy_co2_ustd',
                    'salinity', 'temperature', 'depth')]
        unavg = [('station', 'day', 'bottle_ch4_nM', 'bottle_co2_uatm', 'bottle_depth', 'lat', 'lon',
                    'jy_ch4_ppm', 'jy_ch4_uatm', 'jy_ch4_nm', 'jy_ch4_umolkg', 'jy_co2_ppm', 'jy_co2_uatm',
                    'salinity', 'temperature', 'depth')]
        for i, day in enumerate([28, 29, 30, 1, 2]):
            samples = self.bottle_samples[self.bottle_samples['day'] == day]
            print np.unique(samples['day'].values)
            methane = samples['[CH4] nM'].values
            co2 = samples['pCO2'].values
            lat = samples['lat'].values
            lon = -samples['lon'].values
            depth = samples['depth'].values
            station = samples['station'].values

            if day == 2:
                print(methane, lat, lon)

            jy_df = self.mission[i]
            for j in range(0, len(methane)):
                jy_df.loc[:, 'Distance'] = jy_df.apply(lambda x: get_distance((lat[j], lon[j]), (x['Latitude'], x['Longitude']), geo_epsilon), axis=1)
                if depth[j] == 0.75:
                    d = depth[j]-0.15
                    de = 0.36
                elif depth[j] > 0.75:
                    d = depth[j]
                    de = 0.36
                else:
                    d = depth[j]
                    de = depth_epsilon
                chopped = jy_df[(jy_df['Distance'] == True) &
                                (jy_df['Depth'] <= d + de) &
                                (jy_df['Depth'] >= d - de)]
                print(len(chopped))
                jy_methane = chopped['CH4_ppm'].values
                jy_uatm = chopped['CH4_uatm'].values
                jy_nm = chopped['CH4_nM'].values
                jy_umolkg = chopped['CH4_umolkg'].values
                jy_co2 = chopped['CO2_ppm'].values
                jy_co2_uatm = chopped['CO2_uatm'].values
                jy_salinity = chopped['Salinity'].values
                jy_temperature = chopped['Temperature'].values
                jy_depth = chopped['Depth'].values

                if len(jy_methane) > 1:
                    compare.append((station[j], day, methane[j], co2[j], depth[j], lat[j], lon[j],
                                    np.mean(jy_methane), np.mean(jy_uatm), np.mean(jy_nm), np.mean(jy_umolkg),
                                    np.std(jy_methane), np.std(jy_uatm), np.std(jy_nm), np.std(jy_umolkg),
                                    np.mean(jy_co2), np.mean(jy_co2_uatm),
                                    np.std(jy_co2), np.std(jy_co2_uatm),
                                    np.mean(jy_salinity), np.mean(jy_temperature), np.mean(jy_depth)))
                    for k in range(0,len(jy_methane)):
                        unavg.append((station[j], day, methane[j], co2[j], depth[j], lat[j], lon[j],
                                      jy_methane[k], jy_uatm[k], jy_nm[k], jy_umolkg[k],
                                      jy_co2[k], jy_co2_uatm[k],
                                      jy_salinity[k], jy_temperature[k], jy_depth[k]))

        if save_path is not None:
            np.savetxt(save_path+'bottle_averaged.csv', [x for x in compare], fmt='%s')
            np.savetxt(save_path+'bottle.csv', [x for x in unavg], fmt='%s')

        return compare[1:], unavg[1:]


def clean_samples(filepath):
    '''
    Reads in sample data from filepath and creates data structure
    '''
    samples_df = pd.read_table(filepath, delimiter=',', header=0)
    samples_df['lat'] = pd.to_numeric(samples_df['lat'], errors='coerce')
    samples_df['lon'] = -pd.to_numeric(samples_df['lon'], errors='coerce')
    # samples_df = samples_df.dropna()
    return samples_df

def strip_mission(df, geo_frame='airmar', geo_labels=('lon_mod', 'lat_mod'), meth_eff=0.03, carb_eff=0.70):
    ''' Creates simple frame of the relevant data of interest '''
    print meth_eff
    new_frame = pd.DataFrame()
    new_frame.loc[:, 'CO2_ppm'] = df['gga']['CO2_ppm']
    new_frame.loc[:, 'CO2_uatm'] = df.apply(lambda x: apply_efficiency(x['gga']['CO2_ppm'], eff=carb_eff, gppm=0.), axis=1)

    new_frame.loc[:, 'CH4_ppm'] = df['gga']['CH4_ppm']
    new_frame.loc[:, 'CH4_uatm'] = df.apply(lambda x: apply_efficiency(x['gga']['CH4_ppm'], eff=meth_eff), axis=1)

    new_frame.loc[:, 'CH4_nM'] = df.apply(lambda x: determine_methane(apply_efficiency(x['gga']['CH4_ppm'], eff=meth_eff)*1e-6,
                                                                      x['ctd']['Salinity'],
                                                                      x['ctd']['Temperature'])*1e6, axis=1)
    new_frame.loc[:, 'CH4_umolkg'] = df.apply(lambda x: determine_methane(apply_efficiency(x['gga']['CH4_ppm'], eff=meth_eff)*1e-6,
                                                                          x['ctd']['Salinity'],
                                                                          x['ctd']['Temperature'],
                                                                          units='umolkg')*1e9, axis=1)

    new_frame.loc[:, 'O2'] = df['optode']['O2Concentration']
    new_frame.loc[:, 'Longitude'] = df[geo_frame][geo_labels[0]]
    new_frame.loc[:, 'Latitude'] = df[geo_frame][geo_labels[1]]
    new_frame.loc[:, 'Temperature'] = df['ctd']['Temperature']
    new_frame.loc[:, 'Salinity'] = df['ctd']['Salinity']
    new_frame.loc[:, 'Depth'] = df['ctd']['Depth']
    return new_frame

def determine_methane(fCH4, sal, temp, units='mM'):
    ''' Interfaces with the gasex library to convert to desired units'''
    return sol.sol_SP_pt(sal, temp, gas='CH4', p_dry=fCH4, units=units)

def get_distance(coord1, coord2, limit):
    '''Method to get the distance in meters between two points'''
    try:
        e1, n1, zn1, zl1 = utm.from_latlon(coord1[0], coord1[1])
        e2, n2, zn2, zl2 = utm.from_latlon(coord2[0], -coord2[1])
        dist = (e1-e2)**2 + (n1-n2)**2
        if dist <= limit**2 is True:
            pass
            # print coord1, coord2
        return dist <= limit**2
    except:
        return False

def get_distance_val(coord1, coord2):
    '''Method to get the distance in meters between two points'''
    try:
        e1, n1, zn1, zl1 = utm.from_latlon(coord1[0], coord1[1])
        e2, n2, zn2, zl2 = utm.from_latlon(coord2[0], -coord2[1])
        dist = (e1-e2)**2 + (n1-n2)**2
        return dist
    except:
        return 10000.

def convert_to_utm(coord):
    try:
        e1, n1, zn1, zl1 = utm.from_latlon(coord[0], coord[1])
        return e1, n1
    except:
        return 0., 0.


def apply_efficiency(x, eff=0.15, gppm=1.86):
    '''Method for applying the extraction efficiency'''
    return (x-gppm)/eff + gppm

def convert_CH4(x, eff=0.035, peq=495., gppm=1.86):
    ''' Method to convert the raw ppm measurements from the GGA to compensated
    uatm units '''
    ui = peq * gppm / 1000.
    return (x * peq / 1000. - ui) / eff + ui

def convert_CO2(x, eff=0.70, peq=495., gppm=1.86):
    ''' Method to convert the raw ppm measurements from the GGA to compensated
    uatm units '''
    ui = peq * gppm / 1000.
    return (x * peq / 1000. - ui) / eff + ui