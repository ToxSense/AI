#python3

import numpy as np

import tensorflow as tf

import os

from datetime import datetime

import sqlite3

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import batch_normalization
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.ops.gen_nn_ops import BatchNormWithGlobalNormalization

from calcAQI import officialAQIus


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


##VARS

cwd = os.path.dirname(os.path.realpath(__file__))

#create database
#Connect to DB
sqlCon = sqlite3.connect(cwd + '/trainingDB.db') #connect to sqlite DB-File
sqlCur = sqlCon.cursor() #and create a cursor

tex = False

try: #Create Table if not already
    #timestamp,lat,lon,selfaqi,aqi1,aqi1dist,aqi1dir,aqi2,aqi2dist,aqi2dir,aqi3,aqi3dist,aqi3dir,windspeed,winddir,mapsec
    sqlCur.execute('''CREATE TABLE traindata (place TINYTEXT, start TIMESTAMP, end TIMESTAMP, aqi FLOAT)''')
   
except:
    print('table exists')
    tex = True

if tex == False:
  # load csv content
  csvList = []
  for file in os.listdir(cwd + '/data'):
      if file.endswith(".csv"):
          csvList.append(os.path.join(cwd+'/data', file))
  for csvFile in csvList:
      wFile = open(csvFile, 'r')
      i=0
      for row in wFile:
          if i > 17:
              vals = row.split(",")
              start = datetime.timestamp(datetime.strptime(vals[0], '%d-%m-%Y %H:%M'))
              end = datetime.timestamp(datetime.strptime(vals[1], '%d-%m-%Y %H:%M'))
              pm25 = vals[2].strip('\n')
              if not pm25 == 'None':
                aqi = officialAQIus({'PM10':0.0,'PM25':float(pm25)})
                place = os.path.splitext(os.path.basename(csvFile))[0]
                sqlCur.execute("INSERT INTO traindata VALUES (?,?,?,?)", (place, start, end, aqi))
          i += 1

  sqlCon.commit()





imgList = []
aqiList = []
for files in os.walk(cwd + '/data'):
    for file in files[2]:
        if file.endswith(".jpeg"):
            time = datetime.timestamp(datetime.strptime(os.path.splitext(file)[0], r'%Y-%m-%d %H_%M_%S'))
            place = os.path.basename(os.path.normpath(files[0]))
            sqlCur.execute('SELECT aqi FROM traindata WHERE (place=?) AND (? BETWEEN start AND end)', (place, time))
            sqlAns = sqlCur.fetchone()
            if type(sqlAns) == tuple:
              aqi = sqlAns[0]
            else:
              aqi = None
            imgName = os.path.join(files[0], file)
            if type(aqi) == float:
                aqiList.append(aqi)
                imgList.append(imgName)

sqlCon.close()

print(f"\n\n\nAnzahl AQIs: {len(aqiList)}\n\n\n")

aqiList = list(map(int, aqiList))

#aqiList[:] = [x / 200 for x in aqiList]

a = np.array(aqiList)
b = np.array(imgList)

indices = np.arange(a.shape[0])
np.random.shuffle(indices)

aqiList = a[indices]
imgList = b[indices]

train_imgs = imgList[:round(len(imgList)*1)]
test_imgs = imgList[round(len(imgList)*1):]
train_lbs = aqiList[:round(len(aqiList)*1)]
test_lbs = aqiList[round(len(aqiList)*1):]

print(len(train_imgs),len(test_imgs))


img_height, img_width = 256, 256

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    return img

def process_data(img_path, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    img = decode_img(img)  
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_imgs,train_lbs))
test_ds = tf.data.Dataset.from_tensor_slices((test_imgs,test_lbs))

train_ds = train_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(16)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds)


data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.experimental.preprocessing.RandomZoom((-0.3,0.3)),
  ]
)



""" model = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(200, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
]) """
model = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])
""" model = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])
 """


model.summary()

model.compile(
  optimizer='adam',
  loss="mse", metrics=["mae",]
  )


model.fit(train_ds, epochs=500)
#model.fit(train_ds, validation_data=test_ds, epochs=200)
#model.fit(train_ds, epochs=10)

#predictions_single = model.predict(imgList[30])
#print(predictions_single)

#model.save(cwd + '/saved_models/model' + datetime.now().strftime(r'%Y%m%d_%H%M%S'))


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(cwd + "/" + datetime.now().strftime(r'%Y%m%d_%H%M%S') + 'model.tflite', 'wb') as f:
  f.write(tflite_model)
  f.close()
