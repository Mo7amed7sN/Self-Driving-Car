import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import os
import csv

from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = False

try:
    sys.path.append('G:\CARLA\CARLA_0.9.3\PythonAPI\carla-0.9.3-py3.7-win-amd64.egg')
except IndexError:
    pass

import carla

actor_list = []
IM_WIDTH = 1000
IM_HEIGHT = 1000
model = None
outs = 0.0
def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def preprocess(image):  # preprocess image
    import tensorflow as tf
    return tf.image.resize(image, (200, 66))

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    global model, outs
    img = to_rgb_array(image)
    temp = cv2.resize(img, (320, 160))
    temp = np.expand_dims(temp, axis=0)
    # print(temp.shape)
    outs = model.predict(temp)[0][0]
    outs = round(float(outs), 11) 
    print(outs)
    cv2.waitKey(1)
    return i3/255.0



try:
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(preprocess))
    model.add(Lambda(lambda x: (x / 127.0 - 1.0)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(units=1164, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    
    if os.path.exists('G:\CARLA\CARLA_0.9.3\PilotNet_model\model.hd5'):
    	model.load_weights('G:\CARLA\CARLA_0.9.3\PilotNet_model\model.hd5')

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    bp = blueprint_library.filter("model3")[0]
    print('start of simulation')

    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=outs))
    actor_list.append(vehicle)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")
    
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data:process_img(data))
     
    
    time.sleep(300)
finally:
    for actor in actor_list:
        actor.destroy()
    print("end of simulation")
