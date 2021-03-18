#!/usr/bin/python

#import requests
from PIL import Image
import os
import subprocess
#from xml.dom import minidom
#import pyproj

cwd = os.path.dirname(os.path.realpath(__file__))

class genMaps:
    def __init__(self):
        self.coordinatesList = []
        self.maperiExecPath = r"C:\\Users\\Timo\\Downloads\\Maperitive-latest\\Maperitive\\Maperitive.Console.exe"
        self.maps = {}
    
    def add(self, lat, lon):
        self.coordinatesList.append((lat,lon))

    def generate(self,io=True):
        dlScript = open(cwd + '/maperitive-scripts/dlscript.mscript', 'w')
        i = 0
        self.coordinatesList.sort()
        for co in self.coordinatesList:
            latA = co[0] - 0.001
            latB = co[0] + 0.001
            lonA = co[1] - 0.001
            lonB = co[1] + 0.001

            """ if os.path.isfile(cwd + f'/maperitive-scripts/temp{i}.png'): #check if file exists and is on correct coordinates (then skip)
                georef = minidom.parse(cwd + f'/maperitive-scripts/temp{i}.png.georef')
                x = georef.getElementsByTagName("x")[0].childNodes[0].data
                y = georef.getElementsByTagName("y")[0].childNodes[0].data
                P3857 = pyproj.Proj('epsg:3857')
                P4326 = pyproj.Proj('epsg:4326')
                gLon,gLat = pyproj.transform(P3857, P4326, x, y)
                if latA <= gLat <= latB and lonA <= gLon <= lonB:
                    print('MapFile exists... skipping')
                    i+=1
                    continue """

            dlScript.write(f'set-geo-bounds {lonA},{latA},{lonB},{latB}\ndownload-osm-overpass bounds={lonA},{latA},{lonB},{latB} service-url="http://localhost/api" ' + r'query="way[building]($b$);out;>;out qt;"' + f'\nexport-bitmap file=temp{i}.png width=64 height=64 zoom=16\n')
            i += 1
        dlScript.write('exit')
        dlScript.close()

        constrCmd = [self.maperiExecPath, cwd + '/maperitive-scripts/basicsetting.mscript', cwd + '/maperitive-scripts/dlscript.mscript']
        subprocess.run(constrCmd, cwd=cwd+'/maperitive-scripts/')
        
        i = 0
        for co in self.coordinatesList:
            image = Image.open(cwd + f'/maperitive-scripts/temp{i}.png').convert("L")

            if io == True:
                binarystring = ""

                for x in range(image.width):
                    for y in range(image.height):
                        binarystring += str(round(image.getpixel((x,y)) / 255))

                self.maps.update({(co[0],co[1]):binarystring})
            elif io == False:
                self.maps.update({(co[0],co[1]):image})

            os.remove(cwd + f'/maperitive-scripts/temp{i}.png')
            os.remove(cwd + f'/maperitive-scripts/temp{i}.png.georef')
            i += 1



def testFunc(binarystring):
    testImg = Image.new("L", (64,64))
    i=0
    for x in range(testImg.width):
        for y in range(testImg.height):
            testImg.putpixel( (x,y), int(binarystring[i])*255 )
            i += 1
    testImg.show()

#testFunc(genMap(48.7868,9.2149))