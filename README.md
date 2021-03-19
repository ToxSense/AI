# Learning Atmospheres - ToxSense 

## Artificial Intelligence

### Data Gathering

In this project, the datasets are provided from the sensor.community [sensor.community], an open and free organisation of data, consisting of mostly private people who contribute to the dataset by setting up their own sensors and sharing their measurements to an open API (database access). Due to this organization-concept the ubiquitous sensors generate vast databases, but since it is done by non-professionals, some of the data may be flawed and can not be fully relied on.

Besides these provided datasets, selfmade local collection boxes, with sensors and a new concept for measuring the air quality, were conceived. 

Even if a massive amount of data is preferable for the training of an AI, it implies numerous obstacles in handling and preparing the data for said training, because of the limited computing-power on our hands. The raw data is downloaded in many compressed paquet-table-files, stored on a large-sized hard drive. Due to the size of the files, only one file at a time can be processed, containing one month worth of sensor values. The table is loaded into memory and decompressed to a Pandas DataFrame [Pandas] using the FastParquet library [FastParquet]. This dataframe only contains the raw sensor data, as well as the coordinates of the location and a timestamp. In a first processing step the sensor-data is converted into an AQI, using the officialAQIus library [officialAQIus] from the sensor.community. The next step consists of finding the closest three sensors, to implement their AQIs and relative directions into the training database. The search for the nearest neighbours is a complicated mathematical problem that takes a lot of time and computing resources, thus diminishing the efficiency of the program. Even if a single computing task takes only 20 milliseconds, after only 100 iterations this step takes two seconds. A solution was found in limiting the radius of the coordinates to roughly one kilometer and implementing a parallel multi-threaded calculation, enabling multiple processors to compute at the same time. Then the wind speed and wind direction is fetched from the extensive historical Meteostat database [Meteostat]. Here again, the queries for every sensor location are first bundled into one request to limit the amount of connections, because every connection buildup is taking roughly 10 to 30 ms, whereas the download of the actual, small-sized data is nearly immediate. The final training-database, saved in the sqlite3 format [sqlite3], is now appended by one months worth of the following data: The actual AQI of the selected sensor, the three closest sensors AQI-values with their respective relative directions, and the speed and direction of the wind at the time of recording. In a final step maps for every location in the database sections are generated, using the Maperitive utility [Maperitive]. It connects to an OpenStreetMaps Overpass-API [OpenStreetMaps], providing the required map data, and renders it into a 64 x 64 pixels sized figure ground map. To reduce the request time to the Overpass-API servers and circumvent their limitations, a private Overpass-API-Server [overpass-api] was set up in a virtual environment. The map pixels are then flattened into one bitstring (0 = black, 1 = white) with a length of 4096 bits to be saved into the database as well. This database is used to train the server-AI, as described in the “Data interpretation and results”-section.

The AI working on the smartphone is trained on a database structured with photography and AQI-Label pairs. As an addition to the captured data by the Capture-Box, the open Visionair-dataset [Visionair] was implemented providing a vast number of data.

### Data Interpretation and Results

Both AI are trained on a combined model using a Convolutional-Neural-Network (CNN) for image interpretation and a Multilayer Perceptron (MLP) for the numerical data and its final regression into one single output value. The network employs a mathematical operation called convolution. Convolutional networks are a specialized type of neural networks that use convolution in place of general matrix multiplication. (Goodfellow, 2016) Convolutional neural networks are a promising tool for solving the problem of pattern recognition mainly used on images (Valueva, 2020). On the other hand, a Multilayer Perceptron is composed of multiple Layers of simulated neurons, that are being activated (or deactivated) depending on different mathematical functions processing the input data (Rosenblatt, 1958). The data is run through this model multiple times, increasing its accuracy in each step, by rearranging the neurons activation and the connections between the layers, comparing the estimated value with the actual data.
For the Server-AI, the MLP is trained with numerical data, the three closest aqi values, their direction, as well as the windspeed and direction. The CNNs sole purpose is to interpret a generated map section centered on the requested point with it's surroundings. This part is very valuable as it takes the built environment into consideration.
The MLP consists of two densification layers, one composed of 32 Units, the other of 16. The CNN is created with the shape of the map section, 64 x 64 pixels and one dimension in depth (black & white). Each of three convolutional layers are composed with 16, 32 and 64 filters. The results are then flattened and densed into 32 and 16 Units MLP layers to match the output of the first MLP. A dropout of 0.5 is applied to avoid overtraining. Finally the two models are concatenated into one single MLP with a regression from 32, over 16 to one layer units. All neurons in the MLP and CNN layers are activated with the relu function except the last one that, for regression purposes, is activated linearly. Optimized with an adam-optimizer-function and trained on the squared mean average value, the model always predicts a single AQI-value.
The model for the mobile application is trained with one CNN, similar to the first one with convolutional layers filters with the values of 16, 32 and 64. The regrission is done again with a MLP of 32, 16 and 1 units with similar activation as in the first model. BEfore the data is trained, data-augmentation is applied. This means, that the input images are randomly mirrored, rotated or zoomes, to enhance the later recognition results and diminish the overtraining.



## Credits

This project is imagined and created by Timo Bilhöfer, Markus Pfaff and Maria Rădulescu.

As part of the seminar *Learning Atmospheres* in the winter-semester 2020/21 it is supported by Irina Auernhammer, Silas Kalmbach and Prof. Lucio Blandini from the[ Institute for Lightweight Structures and Conceptual Design (**ILEK**)](https://www.ilek.uni-stuttgart.de/) and is also part of the[ Collaborative Research Centre 1244 (**SFB 1244**)](https://www.sfb1244.uni-stuttgart.de/).



## **Bibliography**

Air Protection and **Climate Policy Office**, Department of Air Quality and Education Monitoring in **Warsaw** (**2020**), Warsaw’s holistic approach to reduce air pollution, https://breathelife2030.org/news/warsaws-holistic-approach-reduce-air-pollution/ Accessed 2021/03/16



**Badach**, Joanna; Voordeckers, Dimitri; Nyka, Lucyna; van Acker, Maarten (**2020**): A framework for Air Quality Management Zones - Useful GIS-based tool for urban planning: Case studies in Antwerp and Gdańsk. In Building and Environment 174, p. 106743. DOI: 10.1016/j.buildenv.2020.106743.



**BreathLife** (**2016**): https://breathelife2030.org/solutions/citywide-solutions/ Accessed 2021/03/16



**Climate & Clean Air Coalition** (**2020**): World Cities day event focuses on how health, climate and urban air pollution are interlinked, https://breathelife2030.org/news/world-cities-day-event-focuses-health-climate-urban-air-pollution-interlinked/ Accessed 2021/03/16



**Das**, Ritwajit (**2020**) How community-based air quality monitoring is helping the city of Bengaluru fight back against air pollution, https://breathelife2030.org/news/community-based-air-quality-monitoring-helping-city-bengaluru-fight-back-air-pollution/ Accessed 2021/03/16

**Goodfellow**, Ian; Bengio, Yoshua; Courville Aaron. (**2016**): *Deep Learning*. MIT Press.

**Institute of hygiene and environment, Hamburg**. Leuchtbakterientest. Accessed 2021/03/17. https://www.hamburg.de/hu/biotestverfahren/2604448/leuchtbakterientest/

**Kang**, Gaganjot Kaur; Gao, Jerry Zeyu; Chiao, Sen; Lu, Shengqiang; Xie, Gang (**2018**): Air Quality Prediction: Big Data and Machine Learning Approaches. In IJESD 9 (1), pp. 8–16. DOI: 10.18178/ijesd.2018.9.1.1066.

**Larson**, Jeff; Mattu, Surya; Kirchner, Lauren; Angwin, Julia (**2016**): How We Analyzed the COMPAS Recidivism Algorithm

**Liao**, Xiong; Tu, Hong; Maddock, Jay E.; Fan, Si; Lan, Guilin; Wu, Yanyan et al. (**2015**): Residents’ perception of air quality, pollution sources, and air pollution control in Nanchang, China. In Atmospheric Pollution Research 6 (5), pp. 835–841. DOI: 10.5094/APR.2015.092.

**Nikolopoulou**, Marialena. (**2009**). PERCEPTION OF AIR POLLUTION AND COMFORT IN THE URBAN, Conference: CISBAT International Scientific Conference, Lausanne

**Nisky**, Ilana; Hartcher-O’Brien, Jess; Wiertlewski, Michaël; Smeets, Jeroen (**2020**): Haptics: Science, Technology, Applications. Cham: Springer International Publishing (12272).

**Peng**, Minggang; Zhang, Hui; Evans, Richard D.; Zhong, Xiaohui; Yang, Kun (**2019**): Actual Air Pollution, Environmental Transparency, and the Perception of Air Pollution in China. In *The Journal of Environment & Development* 28 (1), pp. 78–105. DOI: 10.1177/1070496518821713.

**Rosenblatt**, Frank (**1958**): The perceptron - a probabilistic model for information storage and organization in the brain.

**Smedley**, Tim. **2019**/11/15. The toxic killers in our air too small to see. Accessed 2021/03/14. https://www.bbc.com/future/article/20191113-the-toxic-killers-in-our-air-too-small-to-see

**Sokhanvar**, Saeed S. (**2013**): Tactile sensing and displays. Haptic feedback for minimally invasive surgery and robotics. Chichester, West Sussex, U.K.: John Wiley & Sons.

**Spiroska**, Jana; Rahman, Asif; Pal, Saptarshi (**2011**): Air Pollution in Kolkata: An Analysis of Current Status and Interrelation between Different Factors. In South East European University Review 8 (1). DOI: 10.2478/v10306-012-0012-7.

**Valueva**, M.V.; Nagornov N.N.; Lyakhov, P.A.; Valuev, G.V.; Chervyakov, N.I. (**2020**) Application of the residue number system to reduce hardware costs of the convolutional neural network implementation, Mathematics and Computers in Simulation. https://doi.org/10.1016/j.matcom.2020.04.031.

**VISATON**. **2010**/04. Basic principles of exciter-technology. Accessed 2021/03/14. 

**Vu**, Tuan V.; Shi, Zongbo; Cheng, Jing; Zhang, Qiang; He, Kebin; Wang, Shuxiao; Harrison, Roy M. (**2019**): Assessing the impact of clean air action on air quality trends in Beijing using a machine learning technique. In Atmos. Chem. Phys. 19 (17), pp. 11303–11314. DOI: 10.5194/acp-19-11303-2019.

**World Health Organization (2006)**: Air quality guidelines. Global update 2005 : particulate matter, ozone, nitrogen dioxide, and sulfur dioxide. Copenhagen: World Health Organization.

**Whitney**, Matt; Quin, Hu (**2021**): How China is tackling air pollution with big data, https://breathelife2030.org/news/china-tackling-air-pollution-big-data/ Accessed 2021/03/16

**Wu**, Yi-Chen; Shiledar, Ashutosh; Li, Yi-Cheng; Wong, Jeffrey; Feng, Steve; Chen, Xuan et al. (**2017**): Air quality monitoring using mobile microscopy and machine learning. In Light, science & applications 6 (9), e17046. DOI: 10.1038/lsa.2017.46.



## **Programming resources:**

**1** **Android Bluetooth**. Majdi_la. (Sample Code) https://stackoverflow.com/questions/13450406/how-to-receive-serial-data-using-android-bluetooth. CC BY-SA 3.0.

**2** **Android GPS**. Azhar. (Sample Code) https://www.tutorialspoint.com/how-to-get-the-current-gps-location-programmatically-on-android-using-kotlin. Terms apply.

**3** **Android TFlite**. Anupamchugh. (Sample Code) anupamchugh/AndroidTfLiteCameraX. Pending request.

**4** **FastAPI**. Sebastián Ramírez. (Library) tiangolo/fastapi. https://fastapi.tiangolo.com/. MIT-License.

**5** **I2cdevlib**. Jeff Rowberg. (Library) jrowberg/i2cdevlib. MIT-License.

**6** **Keras: Multiple Inputs and Mixed Data**. Adrian Rosebrock. (Sample Code) https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

**7** **Leaflet**. Vladimir Agafonkin. (Library) Leaflet/Leaflet. https://leafletjs.com/. BSD-2-Clause.

**8** **Maperitive**. Igor Brejc. (Program) https://maperitive.net. Terms apply.

**9** **Meteostat**. Christian Lamprecht. (DB/Library) https://meteostat.net. CC-BY-NC 4.0/MIT-License.

**10** **officialAQIus**. OpenData Stuttgart. Rewritten by Timo Bilhöfer in Python. (Library) https://github.com/opendata-stuttgart/feinstaub-map-v2/blob/master/src/js/feinstaub-api.js. MIT-License.

**11** **OpenStreetMap**. OpenStreetMap contributors. (DB) https://www.openstreetmap.org/copyright. Terms apply.

**12** **Overpass-API**. Wiktorn. (Docker-Image) wiktorn/Overpass-API. AGPL 3.0.

**13** **Pandas**. Pandas contributors. (Library) https://pandas.pydata.org. BSD-3 Clause

**14** **Python**. Python Software Foundation. (Interpreter) https://python.org. PSF-License

**15** **sensor.community**. (DB) https://archive.sensor.community/. Open Data Commons: Database Contents License (DbCL) v1.0.

**16** **Sqlite3**. (Library & DB Language) https://www.sqlite.org. Public Domain.

**17** **TensorFlow**. TensorFlow Community. (Library & Sample Code) https://www.tensorflow.org. Apache-License 2.0.

**18** **VisionAir**. Harshita Diddee, Divyanshu Sharma, Shivam Grover, Shivani Jindal. (DB) https://vision-air.github.io. MIT-License