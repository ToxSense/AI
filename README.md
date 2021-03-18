# Learning Atmospheres - ToxSense 

## Artificial Intelligence

### Data Gathering

In this project, the datasets are provided from the sensor.community [sensor.community], an open and free organisation of data, consisting of mostly private people who contribute to the dataset by setting up their own sensors and sharing their measurements to an open API (database access). Due to this organization-concept the ubiquitous sensors generate vast databases, but since it is done by non-professionals, some of the data may be flawed and can not be fully relied on.

Besides these provided datasets, selfmade local collection boxes, with sensors and a new concept for measuring the air quality, were conceived. 

Even if a massive amount of data is preferable for the training of an AI, it implies numerous obstacles in handling and preparing the data for said training, because of the limited computing-power on our hands. The raw data is downloaded in many compressed paquet-table-files, stored on a large-sized hard drive. Due to the size of the files, only one file at a time can be processed, containing one month worth of sensor values. The table is loaded into memory and decompressed to a Pandas DataFrame [Pandas] using the FastParquet library [FastParquet]. This dataframe only contains the raw sensor data, as well as the coordinates of the location and a timestamp. In a first processing step the sensor-data is converted into an AQI, using the officialAQIus library [officialAQIus] from the sensor.community. The next step consists of finding the closest three sensors, to implement their AQIs and relative directions into the training database. The search for the nearest neighbours is a complicated mathematical problem that takes a lot of time and computing resources, thus diminishing the efficiency of the program. Even if a single computing task takes only 20 milliseconds, after only 100 iterations this step takes two seconds. A solution was found in limiting the radius of the coordinates to roughly one kilometer and implementing a parallel multi-threaded calculation, enabling multiple processors to compute at the same time. Then the wind speed and wind direction is fetched from the extensive historical Meteostat database [Meteostat]. Here again, the queries for every sensor location are first bundled into one request to limit the amount of connections, because every connection buildup is taking roughly 10 to 30 ms, whereas the download of the actual, small-sized data is nearly immediate. The final training-database, saved in the sqlite3 format [sqlite3], is now appended by one months worth of the following data: The actual AQI of the selected sensor, the three closest sensors AQI-values with their respective relative directions, and the speed and direction of the wind at the time of recording. In a final step maps for every location in the database sections are generated, using the Maperitive utility [Maperitive]. It connects to an OpenStreetMaps Overpass-API [OpenStreetMaps], providing the required map data, and renders it into a 64 x 64 pixels sized figure ground map. To reduce the request time to the Overpass-API servers and circumvent their limitations, a private Overpass-API-Server [overpass-api] was set up in a virtual environment. The map pixels are then flattened into one bitstring (0 = black, 1 = white) with a length of 4096 bits to be saved into the database as well. This database is used to train the server-AI, as described in the “Data interpretation and results”-section.

The AI working on the smartphone is trained on a database structured with photography and AQI-Label pairs. As an addition to the captured data by the Capture-Box, the open Visionair-dataset [Visionair] was implemented providing a vast number of data.

### Data Interpretation and Results



## Credits

This project is imagined and created by Timo Bilhöfer, Markus Pfaff and Maria Rădulescu.

As part of the seminar *Learning Atmospheres* in the winter-semester 2020/21 it is supported by Irina Auernhammer, Silas Kalmbach and Prof. Lucio Blandini from the[ Institute for Lightweight Structures and Conceptual Design (**ILEK**)](https://www.ilek.uni-stuttgart.de/) and is also part of the[ Collaborative Research Centre 1244 (**SFB 1244**)](https://www.sfb1244.uni-stuttgart.de/).