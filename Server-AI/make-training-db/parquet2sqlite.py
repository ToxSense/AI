#!/usr/bin/python
import numpy as np
import sqlite3, os, sys
from calcAQI import officialAQIus
from getWind import getWind
from osmDL import genMaps
from datetime import datetime
import pandas as pd
import math
import concurrent.futures

## Variables ##
cwd = os.path.dirname(os.path.realpath(__file__))

## definitions ##
def getCloseAqi(value, coordinates):
    ##get aqi
    aqi = int(officialAQIus({"PM10":value['P1'], "PM25":value['P2']}))

    ##get distance
    lat1 = value['lat']
    lon1 = value['lon']
    lat2 = coordinates[0]
    lon2 = coordinates[1]

    x = lat1 - lat2
    y = lon1 - lon2

    earthRadius=6378137

    xMeters = (x / (180/math.pi)) * earthRadius
    yMeters = (y / (180/math.pi)) * (earthRadius*math.cos(math.pi*x/180))

    aqiDist = int(math.sqrt(xMeters**2 + yMeters**2))

    ##get direction in °
    aqiDir = int(360 * (math.acos(xMeters/(1+aqiDist)))/math.pi)
    if yMeters < 0:
        aqiDir = 360 - aqiDir


    return aqi, aqiDist, aqiDir


def getcStations(vals):

    fromT = vals[2]
    toT = vals[3]
    lat = vals[0]
    lon = vals[1]

    global iWorker, lenSql, data, eta, etaTS
    iWorker += 1

    allData = data

    latO = 0.01
    lonO = 0.01

    if fromT == toT or toT - fromT < 900:
        fromT -= 900
        toT += 900

    #ETA
    if iWorker % 11 == 0 and iWorker > 0:
        eta = round(((lenSql - iWorker) / 11) * (int(datetime.timestamp(datetime.now())) - etaTS) / 60,2)
        etaStatus= round(iWorker / lenSql * 100,2)
        print(f'closest AQI Status: {etaStatus} %\nETA: {eta} mins')
        etaTS = int(datetime.timestamp(datetime.now()))

    closestStations = allData[allData.timestamp.between(np.datetime64(fromT, 's'), np.datetime64(toT, 's')) & allData.lat.between(lat - latO, lat + latO) & allData.lon.between(lon - lonO, lon + lonO) & ((allData.lat != lat) | (allData.lon != lon))]
    closestStations = closestStations.assign(cLat = lat)
    closestStations = closestStations.assign(cLon = lon)
    return closestStations


def addSqlData(sqlFile):
    sqlCon = sqlite3.connect(sqlFile) #connect to sqlite DB-File
    sqlCur = sqlCon.cursor() #and create a cursor
    sqlCur.execute(f'SELECT lat, lon, MIN(timestamp), MAX(timestamp) FROM traindata WHERE aqi1 = 0 AND aqi2 = 0 AND aqi3 = 0 AND windspeed = 0 AND winddir = 0 GROUP BY lat, lon')
    sqlReq = sqlCur.fetchall()

    

    global lenSql, iWorker, eta, etaTS
    lenSql= len(sqlReq)
    print(lenSql)
    iWorker = 0
    eta = 0
    etaTS = int(datetime.timestamp(datetime.now()))
    

    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        stationsResult = executor.map(getcStations, sqlReq)

    allStations = pd.concat(stationsResult)

    i=0

    for vals in sqlReq:
        ###add windspd & dir & mapdata
        i += 1
        print(f"write in DB {i} / {lenSql}")
        
        fromT = vals[2]
        toT = vals[3]
        lat = vals[0]
        lon = vals[1]

        

        cStations = allStations[(allStations['cLat'] == lat) & (allStations['cLon'] == lon)]
        if len(cStations) < 3:
            continue

        dfWind = getWind(fromT, toT, lat, lon)
        if dfWind is None:
            continue
        sqlCur.execute(f'SELECT rowid, timestamp FROM traindata WHERE lat = ? AND lon = ?',(lat,lon))
        allSql = sqlCur.fetchall()
        for row in allSql:
            searchTime = np.datetime64(row[1],'s')
            windVals = dfWind.iloc[(dfWind.index-searchTime).argsort()[:1]]
            if windVals.iloc[0]['wspd'] is None or windVals.iloc[0]['wdir'] is None:
                continue
            
            threeClosest = cStations.iloc[(cStations.timestamp-searchTime).abs().argsort()[:3]]
            
            aqi1, aqi1dist, aqi1dir = getCloseAqi(threeClosest.iloc[0], (lat, lon))
            aqi2, aqi2dist, aqi2dir = getCloseAqi(threeClosest.iloc[1], (lat, lon))
            aqi3, aqi3dist, aqi3dir = getCloseAqi(threeClosest.iloc[2], (lat, lon))
            val = (int(windVals.iloc[0]['wspd']),int(windVals.iloc[0]['wdir']), aqi1, aqi1dist, aqi1dir, aqi2, aqi2dist, aqi2dir, aqi3, aqi3dist, aqi3dir, row[0])
            sqlCur.execute(f"UPDATE traindata SET windspeed = ?, winddir = ?, aqi1 = ?, aqi1dist = ?, aqi1dir = ?, aqi2 = ?, aqi2dist = ?, aqi2dir = ?, aqi3 = ?, aqi3dist = ?, aqi3dir = ? WHERE rowid = ?",val)
            if row[0] % 50 == 0:
                sqlCon.commit()
        
        
    sqlCon.commit()


    ### delete incomplete data
    sqlCur.execute(f"DELETE FROM traindata WHERE aqi1 = 0 OR aqi2 = 0 OR aqi3 = 0 OR selfaqi > 300")

    sqlCon.commit()


    sqlCon.close()
    return (sqlFile)


def generateMaps(sqlFile):
    mapObj = genMaps()
    sqlCon = sqlite3.connect(sqlFile) #connect to sqlite DB-File
    sqlCur = sqlCon.cursor() #and create a cursor
    sqlCur.execute(f'SELECT lat, lon FROM traindata WHERE mapsec = 0 GROUP BY lat, lon')
    sqlReq = sqlCur.fetchall()
    k = 0
    for vals in sqlReq:
        k += 1
        mapObj.add(vals[0], vals[1])
        if k % 100 == 0:
            mapObj.generate()
            for oneMap in mapObj.maps:
                sqlCur.execute(f"UPDATE traindata SET mapsec = ? WHERE lat = ? AND lon = ?", (mapObj.maps[oneMap], oneMap[0], oneMap[1]))
            sqlCon.commit()
            mapObj = genMaps()
    mapObj.generate()
    for oneMap in mapObj.maps:
        sqlCur.execute(f"UPDATE traindata SET mapsec = ? WHERE lat = ? AND lon = ?", (mapObj.maps[oneMap], oneMap[0], oneMap[1]))  
    sqlCon.commit()
    sqlCon.close()
    return sqlFile

def appendTemp2Train(temp, train):
    conTemp = sqlite3.connect(temp) #connect to sqlite DB-File
    curTemp = conTemp.cursor() #and create a cursor
    conTrain = sqlite3.connect(train) #connect to sqlite DB-File
    curTrain = conTrain.cursor() #and create a cursor

    try:
        curTrain.execute(f"CREATE TABLE traindata (timestamp TIMESTAMP, lat FLOAT, lon FLOAT, selfaqi SMALLINT, aqi1 SMALLINT, aqi1dist SMALLINT, aqi1dir SMALLINT, aqi2 SMALLINT, aqi2dist SMALLINT, aqi2dir SMALLINT, aqi3 SMALLINT, aqi3dist SMALLINT, aqi3dir SMALLINT, windspeed SMALLINT, winddir SMALLINT, mapsec BLOB)")
    except:
        print('traintable exists')

    curTemp.execute('SELECT * from traindata')
    traindata = curTemp.fetchall()
    for val in traindata:
        curTrain.execute('INSERT INTO traindata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', val)
    conTrain.commit()
    conTemp.close()
    conTrain.close()
    
    try:
        os.remove(temp)
    except:
     print('Tempfile could not be deleted!!!')

    return train


## MAIN DEF
def p2sql(parquetFiles):
    tempDB = cwd + '/tempDB.db'
    trainDB = cwd + '/trainDB.db'
    for pFile in parquetFiles:
        sqlCon = sqlite3.connect(tempDB) #connect to sqlite DB-File
        sqlCur = sqlCon.cursor() #and create a cursor
        #Connect to DB
        global dirName
        dirName= os.path.split(os.path.split(os.path.split(pFile)[0])[0])[1]
        dirName = 'table' + dirName.replace('-','')
        print(dirName)
        
        global data
        data = pd.read_parquet(pFile, columns=['timestamp', 'lat', 'lon', 'P1', 'P2'], engine="fastparquet")
        data.astype({'timestamp': 'datetime64[s]', 'lat': 'float16','lon': 'float16','P1': 'float16','P2': 'float16',}, copy=False)

        dataSize = len(data)
        targetSize = 10000
        step = int((dataSize / targetSize) -1) #anzahl an datensätzen pro parquet/monat
        print(step)

        #check if data already in table
        testminTS = int(datetime.timestamp(data['timestamp'].min()))
        testmaxTS = int(datetime.timestamp(data['timestamp'].max()))
        trainCon = sqlite3.connect(trainDB) #connect to sqlite DB-File
        trainCur = trainCon.cursor() #and create a cursor
        try:
            trainCur.execute('SELECT MAX(rowid) FROM traindata WHERE timestamp > ? AND timestamp < ?',(testminTS, testmaxTS))
            tsLen = trainCur.fetchone()[0]
            print(tsLen)
            if tsLen > 0:
                print(f'File {pFile} already in trainDB... skipping!')
                continue
        except:
            pass
        trainCon.close()

        try: #Create Table if not already
            #timestamp,lat,lon,selfaqi,aqi1,aqi1dist,aqi1dir,aqi2,aqi2dist,aqi2dir,aqi3,aqi3dist,aqi3dir,windspeed,winddir,mapsec
            sqlCur.execute(f"CREATE TABLE traindata (timestamp TIMESTAMP, lat FLOAT, lon FLOAT, selfaqi SMALLINT, aqi1 SMALLINT, aqi1dist SMALLINT, aqi1dir SMALLINT, aqi2 SMALLINT, aqi2dist SMALLINT, aqi2dir SMALLINT, aqi3 SMALLINT, aqi3dist SMALLINT, aqi3dir SMALLINT, windspeed SMALLINT, winddir SMALLINT, mapsec BLOB)")
        except:
            print('temptable exists, trying to resume...')
            sqlCur.execute('SELECT MAX(rowid) FROM traindata WHERE timestamp > ? AND timestamp < ?',(testminTS, testmaxTS))
            tsLen = sqlCur.fetchone()[0]
            print(tsLen)
            if tsLen > 0:
                print(f'File {pFile} already in tempDB... trying to resume additional Data...')
                addSqlData(tempDB) 
                appendTemp2Train(tempDB, trainDB)
                continue
            else:
                print('Unknown Data... retrying...')
                sqlCon.close()
                os.remove(tempDB)
                sqlCon = sqlite3.connect(tempDB)
                sqlCur = sqlCon.cursor()
                sqlCur.execute(f"CREATE TABLE traindata (timestamp TIMESTAMP, lat FLOAT, lon FLOAT, selfaqi SMALLINT, aqi1 SMALLINT, aqi1dist SMALLINT, aqi1dir SMALLINT, aqi2 SMALLINT, aqi2dist SMALLINT, aqi2dir SMALLINT, aqi3 SMALLINT, aqi3dist SMALLINT, aqi3dir SMALLINT, windspeed SMALLINT, winddir SMALLINT, mapsec BLOB)")


            """
            sqlCon.close()
            os.remove(tempDB)
            sqlCon = sqlite3.connect(tempDB) #connect to sqlite DB-File
            sqlCur = sqlCon.cursor() #and create a cursor
            sqlCur.execute(f"CREATE TABLE traindata (timestamp TIMESTAMP, lat FLOAT, lon FLOAT, selfaqi SMALLINT, aqi1 SMALLINT, aqi1dist SMALLINT, aqi1dir SMALLINT, aqi2 SMALLINT, aqi2dist SMALLINT, aqi2dir SMALLINT, aqi3 SMALLINT, aqi3dist SMALLINT, aqi3dir SMALLINT, windspeed SMALLINT, winddir SMALLINT, mapsec BLOB)")
            """
        
        for index in range(0,dataSize - 1,step): #open parquet file row by row
            row = data.iloc[index]
            timestamp = int(datetime.timestamp(row[0]))
            lat = float(row[1]) #direct2DB
            lon = float(row[2]) #direct2DB
            p10 = float(row[3])
            p25 = float(row[4])

            selfaqi = officialAQIus({"PM10":p10, "PM25":p25}) #2DB

            """ if selfaqi == 0:
                continue """
            
            ##set to 0 for later gen
            aqi1,aqi1dist,aqi1dir,aqi2,aqi2dist,aqi2dir,aqi3,aqi3dist,aqi3dir = (0,0,0,0,0,0,0,0,0)
            winddir = 0
            windspeed = 0
            mapsec = 0
            
            sqlCur.execute(f"INSERT INTO traindata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (timestamp,lat,lon,selfaqi,aqi1,aqi1dist,aqi1dir,aqi2,aqi2dist,aqi2dir,aqi3,aqi3dist,aqi3dir,windspeed,winddir,mapsec))
            if index % (step * 1000) == 0 and index > 0:
                sqlCon.commit()

            sqlCon.commit()
        sqlCon.close()

        addSqlData(tempDB)

        appendTemp2Train(tempDB, trainDB)

 

    generateMaps(trainDB)

    return trainDB


def getParquetFiles(dir):
    fList = []
    for files in os.walk(dir):
        for file in files[2]:
            if file.endswith(".parquet"):
                fList.append(files[0] + '\\' + file)
    return fList


## main program ##
#sqlFile = p2sql([cwd + '/parquetFiles/2019-04/sds011/part-00000-18b6bc00-8426-4900-851e-b650c25b6cee-c000.snappy.parquet'])
print(p2sql(getParquetFiles(cwd + '\\parquetFiles\\')))