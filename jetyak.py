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

class JetYak(object):
    ''' Class which allows for sensor interfaces and basic queries to be made about a jetyak mission '''
    def __init__(self, bounds=None, trim_vals=None):
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
        else:
            print 'Only supporting CTD, GGA, and Airmar inputs \
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
                m.to_csv(save_path+mission_name+'_'+str(i)+'.csv')
                # m.to_csv(save_path+mission_name)

    def load_mission(self, mission_path, header=0, simplify_mission=True):
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
                self.mission.append(strip_mission(temp))

    def add_bottle_samples(self, file_path):
        ''' Method to add bottle samples taken in parallel to the jetyak mission '''
        self.bottle_samples = clean_samples(file_path)

    def extract_bottle_locations(self, geo_epsilon=10.0, depth_epsilon=0.1, save_path=None):
        '''Method to create dataset which matches jetyak and bottle sample information.
        Saves to file is filepath is provided'''
        compare = [('station', 'day', 'bottle_ch4_nM', 'bottle_co2_uatm', 'bottle_depth', 'lat', 'lon',
                    'jy_ch4_ppm', 'jy_ch4_uatm', 'jy_ch4_nm', 'jy_ch4_pstd', 'jy_ch4_ustd', 'jy_ch4_nstd',
                    'jy_co2_ppm', 'jy_co2_uatm', 'jy_co2_pstd', 'jy_co2_ustd',
                    'salinity', 'temperature', 'depth')]
        unavg = [('station', 'day', 'bottle_ch4_nM', 'bottle_co2_uatm', 'bottle_depth', 'lat', 'lon',
                    'jy_ch4_ppm', 'jy_ch4_uatm', 'jy_ch4_nm', 'jy_co2_ppm', 'jy_co2_uatm',
                    'salinity', 'temperature', 'depth')]
        for i, day in enumerate(self.bottle_samples.day.unique()):
            samples = self.bottle_samples[self.bottle_samples['day'] == day]
            methane = samples['[CH4] nM'].values
            co2 = samples['pCO2'].values
            lat = samples['lat'].values
            lon = -samples['lon'].values
            depth = samples['depth'].values
            station = samples['station'].values

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
                jy_methane = chopped['CH4_ppm'].values
                jy_uatm = chopped['CH4_uatm'].values
                jy_nm = chopped['CH4_nM'].values
                jy_co2 = chopped['CO2_ppm'].values
                jy_co2_uatm = chopped['CO2_uatm'].values
                jy_salinity = chopped['Salinity'].values
                jy_temperature = chopped['Temperature'].values
                jy_depth = chopped['Depth'].values

                if len(jy_methane) > 1:
                    compare.append((station[j], day, methane[j], co2[j], depth[j], lat[j], lon[j],
                                    np.mean(jy_methane), np.mean(jy_uatm), np.mean(jy_nm),
                                    np.std(jy_methane), np.std(jy_uatm), np.std(jy_nm),
                                    np.mean(jy_co2), np.mean(jy_co2_uatm),
                                    np.std(jy_co2), np.std(jy_co2_uatm),
                                    np.mean(jy_salinity), np.mean(jy_temperature), np.mean(jy_depth)))
                    for k in range(0,len(jy_methane)):
                        unavg.append((station[j], day, methane[j], co2[j], depth[j], lat[j], lon[j],
                                      jy_methane[k], jy_uatm[k], jy_nm[k],
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

def strip_mission(df, geo_frame='airmar', geo_labels=('lon_mod', 'lat_mod')):
    ''' Creates simple frame of the relevant data of interest '''
    new_frame = pd.DataFrame()
    new_frame.loc[:, 'CO2_ppm'] = df['gga']['CO2_ppm']
    new_frame.loc[:, 'CO2_uatm'] = df['gga']['CO2_uatm']
    new_frame.loc[:, 'CH4_nM'] = df.apply(lambda x: determine_methane(x['gga']['CH4_uatm']/0.000001,
                                                                      x['ctd']['Temperature'],
                                                                      x['ctd']['Salinity'])*0.000001, axis=1)
    new_frame.loc[:, 'CH4_ppm'] = df['gga']['CH4_ppm']
    new_frame.loc[:, 'CH4_uatm'] = df['gga']['CH4_uatm']
    new_frame.loc[:, 'Longitude'] = df[geo_frame][geo_labels[0]]
    new_frame.loc[:, 'Latitude'] = df[geo_frame][geo_labels[1]]
    new_frame.loc[:, 'Temperature'] = df['ctd']['Temperature']
    new_frame.loc[:, 'Salinity'] = df['ctd']['Salinity']
    new_frame.loc[:, 'Depth'] = df['ctd']['Depth']
    return new_frame

def determine_methane(fCH4, sal, temp, units='mM'):
    ''' Corrects from microatms to M, mM or molm3; sal - Practical Salinity, temp - Celsius
    fCH4 - atm'''

    x = sal
    '''Note that salinity argument is Practical Salinity, this is
    beacuse the major ionic components of seawater related to Cl
    are what affect the solubility of non-electrolytes in seawater.'''

    pt68 = temp * 1.00024
    '''pt68 is the potential temperature in degress C on
    the 1968 International Practical Temperature Scale IPTS-68.'''
    K0 = 273.15
    y = pt68 + K0
    y_100 = y * 1e-2

    # Table 1 in Weisenburg and Guinasso 1979
    a = (-68.8862, 101.4956, 28.7314)
    b = (-0.076146, 0.043970, -0.0068672)

    # Bunsen solubility in cc gas @STP / mL H2O atm-1
    CH4_beta = np.exp(a[0] + a[1] * 100/y + a[2] * np.log(y_100) +  x * \
                    (b[0] + b[1] * y_100 + b[2] * y_100**2))
    # Divide by gas virial volume to get mol L-1 atm-1
    CH4sol = CH4_beta / mol_vol(gas='CH4')

    if units == "M":
        return fCH4 * CH4sol[0]
    elif units == "mM" or units == "molm3":
        return fCH4 * CH4sol[0] * 1e3
    else:
        raise ValueError("units: units must be in \'M\'")

def mol_vol(gas=None):
    ''' Accessor for mole-volume coefficients '''
    g_up = gas.upper()
    vol_dict = {'HE':np.array([22.4263]), \
                 'NE':np.array([22.4241]), \
                 'AR':np.array([22.3924]), \
                 'KR':np.array([22.3518]), \
                 'XE':np.array([22.2582]), \
                 'O2':np.array([22.3922]), \
                 'N2':np.array([22.4045]), \
                 'N2O':np.array([22.243]), \
                 'CO2':np.array([0.99498*22.414]), \
                 'CH4':np.array([22.360]), \
                 'H2':np.array([22.428])}
    return vol_dict[g_up]

def get_distance(coord1, coord2, limit):
    '''Method to get the distance in meters between two points'''
    try:
        e1, n1, zn1, zl1 = utm.from_latlon(coord1[0], coord1[1])
        e2, n2, zn2, zl2 = utm.from_latlon(coord2[0], coord2[1])
        dist = (e1-e2)**2 + (n1-n2)**2
        if dist <= limit**2 is True:
            pass
            # print coord1, coord2
        return dist <= limit**2
    except:
        return False
