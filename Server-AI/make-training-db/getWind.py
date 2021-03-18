#!/usr/bin/python

from datetime import datetime
import meteostat as ms
import pandas as pd
import numpy as np
from time import time_ns

##VARS
stationCount = 3

def getWind(timestart, timeend, lat, lon):
    data = pd.DataFrame()

    #ms.Stations.cache_subdir = f'stations{time_ns()}'
    ms.Stations.max_threads = 8
    stations = ms.Stations()
    stations = stations.nearby(lat, lon)
    station = stations.fetch(stationCount)

    
    # Set time period
    if timestart == timeend:
        timestart = timestart - (60 * 30) # - 30 minutes
        timeend = timeend + (60 * 30) # + 30 minutes
    


    start = datetime.fromtimestamp(timestart)
    end = datetime.fromtimestamp(timeend)

    # Get hourly data
    #ms.Hourly.cache_subdir = f'hourly{time_ns()}'
    ms.Hourly.max_threads = 8
    data = ms.Hourly(station, start, end)
    data = data.normalize()
    try:
        data = data.interpolate()
    except:
        pass
    data = data.fetch()

    if len(data) > 0:
        index = 0
        while (data.loc[data.index[index][0]].iloc[0]['wspd'] is None or np.isnan(data.loc[data.index[index][0]].iloc[0]['wspd']) or data.loc[data.index[index][0]].iloc[0]['wdir'] is None or np.isnan(data.loc[data.index[index][0]].iloc[0]['wdir'])):
            index +=1
            if index == 3:
                return None
        return data.loc[data.index[index][0]][['wspd', 'wdir']]

    return None