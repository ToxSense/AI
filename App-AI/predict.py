#python3

import numpy as np

import tensorflow as tf

import os

import sys


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


##VARS

cwd = os.path.dirname(os.path.realpath(__file__))
loadModel = cwd + '/saved_models\model20210125_131310'
img2test = cwd + '/testimg4.jpeg'
imgsizeModel = [256,256]

def decode_img(img_path):
    img = tf.io.read_file(img_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    img = tf.image.resize(img, imgsizeModel)
    return img


img = tf.keras.preprocessing.image.load_img(
    img2test, target_size=tuple(imgsizeModel)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

""" 
model = tf.keras.models.load_model(loadModel)
#model.summary()
predictions = model.predict(img_array)
print(predictions)
"""



# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=cwd+"\\20210212_113236model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_data = img_array
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


""" score = tf.nn.softmax(predictions[0])

aqis = list(np.argpartition(score, -4)[-4:])
confs = list(100 * np.partition(score, -4)[-4:])


result = np.argmax(score)

print(f"AQIs: {aqis}\nScores: {confs}")

print(
    "This image most likely has an AQI of {} with a {:.2f} percent confidence."
    .format(int(result), 100 * np.max(score))
) """