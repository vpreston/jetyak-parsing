import netCDF4
import pandas as pd

fpath = '/home/vpreston/Downloads/'
fname = 'ZCYL5_20180915v30001.nc'

nc = netCDF4.Dataset(fpath+fname, mode='r')

var = nc.variables.keys()
time = nc.variables['time']
dtime = netCDF4.num2date(time[:],time.units)

df = pd.DataFrame()

for v in var:
	try:
		temp = nc.variables[v][:]
		df.loc[:, v] = temp
		# df = pd.Series(temp, index=dtime)
		# df.to_csv(fname+'_'+v+'.csv')
	except:
		print v

print df.head()
df.to_csv(fname+'_converted.csv', header=True)

