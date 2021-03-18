#!/usr/bin/python

import tensorflow as tf
import os
import pandas as pd
from datetime import datetime
import sqlite3
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


## Variables ##
cwd = os.path.dirname(os.path.realpath(__file__))

""" def createImg(binarystring):
    img = Image.new("L", (64,64))
    i=0
    for x in range(img.width):
        for y in range(img.height):
            img.putpixel( (x,y), int(binarystring[i])*255 )
            i += 1
    return img """
def createImg(df):
    imgs = []
    for index, value in df.iteritems():
        img = np.zeros((64, 64, 1), dtype="uint8")
        j=0
        for x in range(64):
            for y in range(64):
                img[x,y] = int(value[j])
                j += 1
        imgs.append(img)
    return np.array(imgs)

def create_mlp(dim, regress=False):
    # define our MLP network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=dim, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    if regress:
        model.add(tf.keras.layers.Dense(1, activation="linear"))
    return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = tf.keras.Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = tf.keras.layers.Conv2D(f, 3, padding="same")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
        # flatten the volume, then FC => RELU => BN => DROPOUT
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = tf.keras.layers.Dense(1, activation="linear")(x)
    # construct the CNN
    model = tf.keras.models.Model(inputs, x)
    # return the CNN
    return model

def cMinMax(df):
    df[['aqi1', 'aqi2', 'aqi3']] = df[['aqi1', 'aqi2', 'aqi3']] / 300.00
    #distMax = sorted([df['aqi1dist'].max(), df['aqi2dist'].max(), df['aqi3dist'].max()])[2]
    #df[['aqi1dist', 'aqi2dist', 'aqi3dist']] = df[['aqi1dist', 'aqi2dist', 'aqi3dist']] / distMax
    df[['aqi1dir', 'aqi2dir', 'aqi3dir', 'winddir']] = df[['aqi1dir', 'aqi2dir', 'aqi3dir', 'winddir']] / 360.00
    df['windspeed'] = df['windspeed'] / 220
    return df

def configure_for_performance(dsL):
    for ds in dsL:
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(16)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dsL


con = sqlite3.connect( cwd + '/trainDB.db')

df = pd.read_sql('SELECT selfaqi,aqi1,aqi1dir,aqi2,aqi2dir,aqi3,aqi3dir,windspeed,winddir,mapsec FROM traindata limit 600', con)
#df = pd.read_sql('SELECT traindata', con, columns=['selfaqi','aqi1','aqi1dist','aqi1dir','aqi2','aqi2dist','aqi2dir','aqi3','aqi3dist','aqi3dir','windspeed', 'winddir', 'mapsec'])
df = df.astype({'selfaqi': 'int16','aqi1': 'int16','aqi1dir': 'int16','aqi3': 'int16','aqi3dir': 'int16','aqi2': 'int16','aqi2dir': 'int16','windspeed': 'int16','winddir': 'int16', 'mapsec':'string'},)

target = df.pop('selfaqi')
#target = df.pop('selfaqi')
print('converting mapsecbytes to imgs...')
mapsec = createImg(df.pop('mapsec'))
print('FINISHED\n')

#print(df)
print('MinMax Data...')
df = cMinMax(df)
print('FINISHED\n')

print(df)
""" dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
train_dataset = dataset.shuffle(len(df)).batch(1024) """



split = train_test_split(df, mapsec, target, test_size=0.1, random_state=42)
(trainX, testX, trainMap, testMap, trainTarget, testTarget) = split

""" trainds = tf.data.Dataset.from_tensor_slices(([trainX, trainMap], trainTarget))
testds = tf.data.Dataset.from_tensor_slices(([testX, testMap], testTarget))

(trainds, testds) = configure_for_performance((trainds, testds)) """

print((trainX, testX, trainMap, testMap, trainTarget, testTarget))




# create the MLP and CNN models
mlp = create_mlp(df.shape[1], regress=False)
cnn = create_cnn(64, 64, 1, regress=False)
# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = tf.keras.layers.Dense(16, activation="relu")(combinedInput)
x = tf.keras.layers.Dense(1, activation="linear")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = tf.keras.models.Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mse", metrics=["mae",], optimizer=opt)

model.fit(x=[trainX, trainMap], y=trainTarget, validation_data=([testX, testMap], testTarget), epochs=100, batch_size=16)

model.save(cwd + '/saved_models/model' + datetime.now().strftime(r'%Y%m%d_%H%M%S'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(cwd + "/" + datetime.now().strftime(r'%Y%m%d_%H%M%S') + 'model.tflite', 'wb') as f:
  f.write(tflite_model)
  f.close()


##TESTING


print("[INFO] predicting aqi...")
preds = model.predict([testX, testMap])

print(preds)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testTarget
percentDiff = (diff / testTarget) * 100
absPercentDiff = np.abs(percentDiff)
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
# finally, show some statistics on our model
""" print("[INFO] avg. aqi: {}, std aqi: {}".format(
	target["selfaqi"].mean(), grouping=True),
	target["selfaqi"].std(), grouping=True)) """
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
